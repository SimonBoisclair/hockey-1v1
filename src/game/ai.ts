import { GameState, Vec2 } from './types';
import {
  RINK_WIDTH, RINK_HEIGHT, PLAYER_RADIUS,
  BACKCHECK_DEPTH, GOAL_DEPTH,
  STEAL_MAX_DIST, STEAL_MIN_DIST, STEAL_CHANCE_FAR, STEAL_CHANCE_NEAR,
  STEAL_DIR_BONUS, STEAL_DIR_PENALTY, MIN_SPEED,
} from './constants';
import { v2, v2Distance, v2Length, v2Normalize, v2Sub, v2Dot } from './physics';

// ── Matrix / vector utilities ──

function randomWeight(fanIn: number): number {
  return (Math.random() * 2 - 1) * Math.sqrt(2 / fanIn);
}

function matVecMul(m: number[][], v: number[]): number[] {
  const rows = m.length;
  const cols = v.length;
  const out = new Array(rows);
  for (let i = 0; i < rows; i++) {
    let s = 0;
    const row = m[i];
    for (let j = 0; j < cols; j++) s += row[j] * v[j];
    out[i] = s;
  }
  return out;
}

function vecAdd(a: number[], b: number[]): number[] {
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

function relu(v: number[]): number[] {
  const out = new Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] > 0 ? v[i] : 0;
  return out;
}

function softmax(v: number[]): number[] {
  const max = Math.max(...v);
  const exp = new Array(v.length);
  let sum = 0;
  for (let i = 0; i < v.length; i++) {
    exp[i] = Math.exp(v[i] - max);
    sum += exp[i];
  }
  for (let i = 0; i < v.length; i++) exp[i] /= sum;
  return exp;
}

// ── MLP Policy Network ──

interface Layer {
  weights: number[][];
  biases: number[];
}

export interface ForwardCache {
  inputs: number[][];
  preAct: number[][];
  postAct: number[][];
}

export const NUM_FEATURES = 20;
export const NUM_ACTIONS = 10;
export const NETWORK_SIZES = [NUM_FEATURES, 512, 512, 256, 128, NUM_ACTIONS];

const STEAL_BUDGET = 5; // matches DECISION_BUDGET

export class PolicyNetwork {
  layers: Layer[];

  constructor(sizes: number[] = NETWORK_SIZES) {
    this.layers = [];
    for (let i = 0; i < sizes.length - 1; i++) {
      const rows = sizes[i + 1];
      const cols = sizes[i];
      const w: number[][] = [];
      for (let r = 0; r < rows; r++) {
        w[r] = new Array(cols);
        for (let c = 0; c < cols; c++) w[r][c] = randomWeight(cols);
      }
      this.layers.push({ weights: w, biases: new Array(rows).fill(0) });
    }
  }

  forward(input: number[]): { probs: number[]; cache: ForwardCache } {
    const cache: ForwardCache = { inputs: [input], preAct: [], postAct: [] };
    let cur = input;
    for (let l = 0; l < this.layers.length; l++) {
      const z = vecAdd(matVecMul(this.layers[l].weights, cur), this.layers[l].biases);
      cache.preAct.push(z);
      cur = l < this.layers.length - 1 ? relu(z) : softmax(z);
      cache.postAct.push(cur);
      if (l < this.layers.length - 1) cache.inputs.push(cur);
    }
    return { probs: cur, cache };
  }

  update(cache: ForwardCache, action: number, advantage: number, lr: number): void {
    const probs = cache.postAct[cache.postAct.length - 1];
    let delta = new Array(probs.length);
    for (let i = 0; i < probs.length; i++) {
      delta[i] = (i === action ? 1 - probs[i] : -probs[i]) * advantage;
    }

    for (let l = this.layers.length - 1; l >= 0; l--) {
      const inp = cache.inputs[l];
      const layer = this.layers[l];

      for (let i = 0; i < layer.weights.length; i++) {
        for (let j = 0; j < layer.weights[i].length; j++) {
          layer.weights[i][j] += lr * delta[i] * inp[j];
        }
        layer.biases[i] += lr * delta[i];
      }

      if (l > 0) {
        const prev = new Array(inp.length).fill(0);
        for (let j = 0; j < inp.length; j++) {
          for (let i = 0; i < delta.length; i++) {
            prev[j] += layer.weights[i][j] * delta[i];
          }
          if (cache.preAct[l - 1][j] <= 0) prev[j] = 0;
        }
        delta = prev;
      }
    }
  }

  clone(): PolicyNetwork {
    const net = new PolicyNetwork([1, 1]);
    net.layers = this.layers.map(l => ({
      weights: l.weights.map(r => [...r]),
      biases: [...l.biases],
    }));
    return net;
  }

  serialize(): string {
    return JSON.stringify(this.layers);
  }

  static deserialize(json: string): PolicyNetwork {
    const net = new PolicyNetwork([1, 1]);
    net.layers = JSON.parse(json);
    return net;
  }
}

// ── State encoding (from acting player's perspective) ──

function computeStealChance(state: GameState, playerIdx: number): number {
  if (state.possession === playerIdx) return 0; // already have puck
  const me = state.players[playerIdx];
  const carrier = state.players[state.possession];
  const dist = v2Distance(me.pos, carrier.pos);
  if (dist > STEAL_MAX_DIST) return 0;
  if (me.stealLockTimer > 0) return 0;

  let chance: number;
  if (dist <= STEAL_MIN_DIST) {
    chance = STEAL_CHANCE_NEAR;
  } else {
    const t = (dist - STEAL_MIN_DIST) / (STEAL_MAX_DIST - STEAL_MIN_DIST);
    chance = STEAL_CHANCE_NEAR + t * (STEAL_CHANCE_FAR - STEAL_CHANCE_NEAR);
  }

  const carrierSpeed = v2Length(carrier.vel);
  if (carrierSpeed > MIN_SPEED) {
    const toStealer = v2Normalize(v2Sub(me.pos, carrier.pos));
    const carrierDir = v2Normalize(carrier.vel);
    const dot = v2Dot(carrierDir, toStealer);
    if (dot > 0) {
      chance += dot * STEAL_DIR_BONUS;
    } else {
      chance += dot * STEAL_DIR_PENALTY;
    }
    chance = Math.max(0.05, Math.min(0.85, chance));
  }
  return chance;
}

export function encodeState(state: GameState, playerIdx: number, decisionTokens: number = STEAL_BUDGET): number[] {
  const me = state.players[playerIdx];
  const opp = state.players[1 - playerIdx];
  const hasPuck = state.possession === playerIdx;
  const goalX = RINK_WIDTH - GOAL_DEPTH / 2;
  const goalY = RINK_HEIGHT / 2;

  return [
    me.pos.x / RINK_WIDTH,
    me.pos.y / RINK_HEIGHT,
    opp.pos.x / RINK_WIDTH,
    opp.pos.y / RINK_HEIGHT,
    me.vel.x / 14,
    me.vel.y / 14,
    opp.vel.x / 14,
    opp.vel.y / 14,
    hasPuck ? 1 : 0,
    me.mustBackcheck ? 1 : 0,
    (goalX - me.pos.x) / RINK_WIDTH,
    (goalY - me.pos.y) / RINK_HEIGHT,
    v2Distance(me.pos, opp.pos) / RINK_WIDTH,
    // New features
    me.destination ? me.destination.x / RINK_WIDTH : me.pos.x / RINK_WIDTH,
    me.destination ? me.destination.y / RINK_HEIGHT : me.pos.y / RINK_HEIGHT,
    decisionTokens / STEAL_BUDGET,
    me.lockDirection ? 1 : 0,
    computeStealChance(state, playerIdx),
    hasPuck ? 0 : 1, // opp has puck
    opp.lockDirection ? 1 : 0,
  ];
}

// ── Action decoding ──

const MOVE_DIST = 12;

export function decodeAction(
  action: number,
  state: GameState,
  playerIdx: number,
): { destination: Vec2 | null; steal: boolean } {
  const me = state.players[playerIdx];
  const opp = state.players[1 - playerIdx];
  const r = PLAYER_RADIUS;

  const clamp = (p: Vec2): Vec2 => ({
    x: Math.max(r, Math.min(RINK_WIDTH - r, p.x)),
    y: Math.max(r, Math.min(RINK_HEIGHT - r, p.y)),
  });

  switch (action) {
    case 0: return { destination: clamp(v2(RINK_WIDTH - GOAL_DEPTH / 2, RINK_HEIGHT / 2)), steal: false };
    case 1: return { destination: clamp(v2(opp.pos.x, opp.pos.y)), steal: false };
    case 2: return { destination: clamp(v2(BACKCHECK_DEPTH / 2, RINK_HEIGHT / 2)), steal: false };
    case 3: return { destination: clamp(v2(me.pos.x, me.pos.y - MOVE_DIST)), steal: false };
    case 4: return { destination: clamp(v2(me.pos.x, me.pos.y + MOVE_DIST)), steal: false };
    case 5: return { destination: clamp(v2(me.pos.x + MOVE_DIST, me.pos.y - MOVE_DIST * 0.7)), steal: false };
    case 6: return { destination: clamp(v2(me.pos.x + MOVE_DIST, me.pos.y + MOVE_DIST * 0.7)), steal: false };
    case 7: return { destination: clamp(v2(me.pos.x - MOVE_DIST, me.pos.y)), steal: false };
    case 8: return { destination: null, steal: true };
    case 9:
    default: return { destination: null, steal: false };
  }
}

// ── Trajectory step ──

export interface TrajectoryStep {
  features: number[];
  action: number;
  cache: ForwardCache;
  reward: number;
}

// ── AI Agent ──

// Decision budget: 5 decisions per 10-second window (600 frames at 60fps).
// The AI can burst decisions (e.g. 3 in 1 second) but then must wait.
const DECISION_BUDGET = 5;
const BUDGET_WINDOW = 600; // frames (10 seconds at 60fps)
const BUDGET_REFILL_RATE = DECISION_BUDGET / BUDGET_WINDOW; // tokens per frame

export class AIAgent {
  network: PolicyNetwork;
  trajectory: TrajectoryStep[];
  lastAction: number;
  explorationRate: number;
  decisionTokens: number; // current budget tokens available
  stepsSinceLastDecision: number;

  constructor(network: PolicyNetwork) {
    this.network = network;
    this.trajectory = [];
    this.lastAction = 9;
    this.explorationRate = 0.1;
    this.decisionTokens = DECISION_BUDGET;
    this.stepsSinceLastDecision = 0;
  }

  act(state: GameState, playerIdx: number): { destination: Vec2 | null; steal: boolean } {
    // Refill tokens each frame (capped at max budget)
    this.decisionTokens = Math.min(DECISION_BUDGET, this.decisionTokens + BUDGET_REFILL_RATE);
    this.stepsSinceLastDecision++;

    // Need at least 1 token to make a new decision
    if (this.decisionTokens < 1) {
      return decodeAction(this.lastAction, state, playerIdx);
    }

    // Must wait at least 1 frame between decisions
    if (this.stepsSinceLastDecision < 2) {
      return decodeAction(this.lastAction, state, playerIdx);
    }

    // Spend a token and make a decision
    this.decisionTokens -= 1;
    this.stepsSinceLastDecision = 0;

    const features = encodeState(state, playerIdx, this.decisionTokens);
    const { probs, cache } = this.network.forward(features);

    let action: number;
    if (Math.random() < this.explorationRate) {
      action = Math.floor(Math.random() * NUM_ACTIONS);
    } else {
      let r = Math.random();
      action = NUM_ACTIONS - 1;
      for (let i = 0; i < NUM_ACTIONS; i++) {
        r -= probs[i];
        if (r <= 0) { action = i; break; }
      }
    }

    this.lastAction = action;
    this.trajectory.push({ features, action, cache, reward: 0 });
    return decodeAction(action, state, playerIdx);
  }

  reset(): void {
    this.trajectory = [];
    this.lastAction = 9;
    this.decisionTokens = DECISION_BUDGET;
    this.stepsSinceLastDecision = 0;
  }
}

// ── REINFORCE training ──

export function computeReturns(rewards: number[], gamma: number): number[] {
  const R = new Array(rewards.length);
  let G = 0;
  for (let i = rewards.length - 1; i >= 0; i--) {
    G = rewards[i] + gamma * G;
    R[i] = G;
  }
  return R;
}

export function trainAgent(agent: AIAgent, gamma: number, lr: number): void {
  const traj = agent.trajectory;
  if (traj.length === 0) return;

  const rewards = traj.map(s => s.reward);
  const returns = computeReturns(rewards, gamma);

  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((a, b) => a + (b - mean) ** 2, 0) / returns.length;
  const std = Math.sqrt(variance) + 1e-8;

  for (let i = 0; i < traj.length; i++) {
    const advantage = (returns[i] - mean) / std;
    agent.network.update(traj[i].cache, traj[i].action, advantage, lr);
  }
}
