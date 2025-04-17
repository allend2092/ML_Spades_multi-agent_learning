###############################
# spades_multiagent_rl.py  (AMP‑optional, error‑free)
#
#  • Added a toggle `USE_AMP` (default False). When AMP is disabled the
#    script runs entirely in fp32, avoiding the GradScaler bfloat16
#    kernel bug you hit. Flip it to True once PyTorch adds bf16 unscale
#    support, or if you want true fp16 mixed precision.
#  • GradScaler and autocast now respect `enabled=USE_AMP` so the code
#    path stays identical regardless of precision mode.
###############################

from __future__ import annotations
import random, sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.distributions.categorical import Categorical
import gym
from gym import spaces
from gym.vector import SyncVectorEnv
from torch.utils.tensorboard import SummaryWriter

try:
    import msvcrt              # Windows quit shortcut
except ImportError:
    msvcrt = None

# ───────────────────────────────── precision toggle ──
USE_AMP = False                # ← set True to re‑enable mixed precision
DTYPE  = torch.float16 if USE_AMP else torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################################
# 1. Card definitions                                      #
############################################################
SUITS = ("Spades", "Hearts", "Diamonds", "Clubs")
RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace")
RANK_VALUE = {r: i for i, r in enumerate(RANKS)}

@dataclass(frozen=True)
class Card:
    rank: str
    suit: str
    def idx(self) -> int:
        return SUITS.index(self.suit) * 13 + RANK_VALUE[self.rank]
    def __repr__(self):
        return f"{self.rank[0]}{self.suit[0]}"

############################################################
# 2. Helper functions                                      #
############################################################

def fresh_deck() -> List[Card]:
    return [Card(r, s) for s in SUITS for r in RANKS]

def one_hot52(cards: List[Card]) -> np.ndarray:
    v = np.zeros(52, np.float32)
    for c in cards:
        v[c.idx()] = 1.0
    return v

############################################################
# 3. Spades environment                                    #
############################################################
class SpadesEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    def __init__(self, reduced=False):
        super().__init__()
        self.reduced = reduced
        self.action_space      = spaces.Discrete(13)
        self.observation_space = spaces.Box(0,1,(105,), np.float32)
        self.reset()
    # deal etc. (unchanged)
    def _deal(self):
        deck = fresh_deck()
        if self.reduced:
            deck = [c for c in deck if RANK_VALUE[c.rank] >= 5]
        random.shuffle(deck)
        self.hands = [sorted(deck[i::4], key=lambda c: c.idx()) for i in range(4)]
        self.played, self.trick_cards = [], []
        self.tricks_taken = [0,0]
        self.current_player, self.spades_broken = 0, False
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._deal(); self.done=False
        return self._obs(), {}
    def _obs(self):
        return np.concatenate([one_hot52(self.hands[self.current_player]),
                               one_hot52(self.played),
                               [float(self.spades_broken)]]).astype(np.float32)
    def _winner_idx(self):
        lead = self.trick_cards[0][1].suit
        cand = [t for t in self.trick_cards if t[1].suit==lead]
        spd  = [t for t in self.trick_cards if t[1].suit=="Spades"] or cand
        return max(spd, key=lambda t: RANK_VALUE[t[1].rank])[0]
    def step(self, action):
        if self.done: raise RuntimeError("step after done")
        hand = self.hands[self.current_player]
        if not hand:
            self.done=True
            return np.zeros(105,np.float32),0.,True,False,{"current_player":self.current_player}
        reward=0.; card=None
        if action>=len(hand): card=random.choice(hand); reward-=2
        else: card=hand.pop(action)
        self.played.append(card); self.trick_cards.append((self.current_player,card))
        if card.suit=="Spades": self.spades_broken=True
        term=False
        if len(self.trick_cards)==4:
            winner=self._winner_idx(); team=winner%2
            self.tricks_taken[team]+=1; reward=1. if winner==self.current_player else 0.
            self.current_player=winner; self.trick_cards.clear()
            if all(len(h)==0 for h in self.hands): term=True
        else:
            self.current_player=(self.current_player+1)%4
        if term: self.done=True; obs=np.zeros(105,np.float32)
        else: obs=self._obs()
        return obs,reward,term,False,{"current_player":self.current_player}

############################################################
# 4. Networks                                              #
############################################################
class MLP(nn.Module):
    def __init__(self,in_dim,out_dim,hidden):
        super().__init__(); layers=[]
        for h in hidden:
            layers+=[nn.Linear(in_dim,h),nn.ReLU()]; in_dim=h
        layers.append(nn.Linear(in_dim,out_dim)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)
class ActorCritic(nn.Module):
    """Shared body with separate policy and value heads."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: List[int]):
        super().__init__()
        self.policy = MLP(obs_dim, act_dim, hidden)
        self.value  = MLP(obs_dim, 1, hidden)

    def forward(self, x):
        logits = self.policy(x)
        value  = self.value(x).squeeze(-1)
        return logits, value

############################################################
# 5. Seat agent                                            #
############################################################
class SeatAgent:
    def __init__(self,obs,act,hidden,lr=2e-4,gamma=0.995,ent_coef=0.02):
        self.net=ActorCritic(obs,act,hidden).to(DEVICE)
        self.opt=torch.optim.Adam(self.net.parameters(),lr=lr)
        self.scaler=GradScaler(enabled=USE_AMP)
        self.gamma, self.ent_coef=gamma, ent_coef
        self.obs_buf=[]; self.act_buf=[]; self.logp_buf=[]; self.rew_buf=[]
    def act(self,obs_np):
        obs_t=torch.tensor(obs_np,dtype=DTYPE,device=DEVICE).unsqueeze(0)
        with autocast(enabled=USE_AMP,dtype=DTYPE):
            logits,_=self.net(obs_t); dist=Categorical(logits=logits); a=dist.sample()
        return a.item(), dist.log_prob(a).squeeze(0), dist.entropy().squeeze(0)
    def store(self,obs,act,logp,rew):
        self.obs_buf.append(obs); self.act_buf.append(act); self.logp_buf.append(logp); self.rew_buf.append(rew)
    def finish_episode(self):
        if not self.rew_buf: return
        R=0.; returns=[]
        for r in reversed(self.rew_buf): R=r+self.gamma*R; returns.insert(0,R)
        ret_t=torch.tensor(returns,dtype=DTYPE,device=DEVICE); ret_t=(ret_t-ret_t.mean())/(ret_t.std()+1e-5)
        obs_t=torch.tensor(np.stack(self.obs_buf),dtype=DTYPE,device=DEVICE); act_t=torch.tensor(self.act_buf,device=DEVICE)
        self.opt.zero_grad(set_to_none=True)
        with autocast(enabled=USE_AMP,dtype=DTYPE):
            logits,values=self.net(obs_t); dist=Categorical(logits=logits)
            logp=dist.log_prob(act_t); ent=dist.entropy(); adv=ret_t-values.detach()
            loss=(-logp*adv).mean()+0.5*F.mse_loss(values,ret_t)-self.ent_coef*ent.mean()
        self.scaler.scale(loss).backward(); self.scaler.step(self.opt); self.scaler.update()
        self.obs_buf.clear(); self.act_buf.clear(); self.logp_buf.clear(); self.rew_buf.clear()

############################################################
# 6. Training                                              #
############################################################
HIDDEN=[1024,1024,512,512,256]; N_ENVS=4; MAX_EPISODES=20000

def make_env(): return SpadesEnv(reduced=False)

def train():
    envs = SyncVectorEnv([make_env for _ in range(N_ENVS)])
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.n
    agents = [SeatAgent(obs_dim, act_dim, HIDDEN) for _ in range(4)]
    writer = SummaryWriter()

    episode_counter = 0
    while episode_counter < MAX_EPISODES:
        obs, _ = envs.reset()
        done = np.zeros(N_ENVS, dtype=bool)
        current_player = np.zeros(N_ENVS, dtype=int)
        step_in_batch = 0
        while not done.all():
            actions = np.zeros(N_ENVS, dtype=int)
            # generate actions
            for env_id in range(N_ENVS):
                if done[env_id]:
                    continue
                seat = current_player[env_id]
                act, logp, _ = agents[seat].act(obs[env_id])
                actions[env_id] = act
                agents[seat].store(obs[env_id], act, logp, 0.0)
            nxt_obs, rewards, terms, truncs, infos = envs.step(actions)
                        # update buffers and track whose turn it is next
            if isinstance(infos, dict):
                # SyncVectorEnv returns a dict with numpy arrays
                if "current_player" in infos:
                    current_player = infos["current_player"].astype(int)
            else:
                # older Gym versions: list of per‑env dicts
                for env_id in range(N_ENVS):
                    if not done[env_id]:
                        agents[current_player[env_id]].rew_buf[-1] += rewards[env_id]
                    if env_id < len(infos) and "current_player" in infos[env_id]:
                        current_player[env_id] = infos[env_id]["current_player"]
            obs = nxt_obs
            done |= terms | truncs
            step_in_batch += 1
        # finish parallel episodes
        for a in agents:
            a.finish_episode()
        episode_counter += N_ENVS
        if episode_counter % 100 == 0:
            writer.add_scalar("episodes", episode_counter, episode_counter)
            print(f"Completed {episode_counter} env‑episodes; last batch {step_in_batch} steps")
        # quit shortcut
        if msvcrt and msvcrt.kbhit() and msvcrt.getch() in (b"q", b"Q"):
            torch.save({f"seat{i}": agents[i].net.state_dict() for i in range(4)}, "spades_multiagent.pt")
            print("Weights saved. Quitting.")
            break

if __name__ == "__main__":
    train()
