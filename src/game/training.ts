import { GameState } from './types';
import {
  createInitialState, gameStep, attemptSteal,
} from './physics';
import { FIXED_DT, RINK_WIDTH } from './constants';
import { AIAgent, trainAgent, PolicyNetwork, NETWORK_SIZES } from './ai';

// Inactivity penalty: penalize agents that don't decide for 5+ seconds
const INACTIVITY_THRESHOLD = 300; // 5 seconds at 60fps
const INACTIVITY_PENALTY = -0.001;

// ── Training stats ──

export interface EpisodeResult {
  blue: number;
  red: number;
}

export interface TrainingStats {
  episode: number;
  blueWins: number;
  redWins: number;
  draws: number;
  winRate: number;
  recentScores: EpisodeResult[];
  avgGoalsPerEp: number;
}

// ── Training config ──

export interface TrainingConfig {
  maxSteps: number;
  gamma: number;
  lr: number;
}

const DEFAULTS: TrainingConfig = {
  maxSteps: 900,   // 15 seconds at 60fps
  gamma: 0.99,
  lr: 0.003,
};

// ── Trainer ──

export class Trainer {
  network: PolicyNetwork;
  agent0: AIAgent;
  agent1: AIAgent;
  stats: TrainingStats;
  config: TrainingConfig;
  recentResults: number[]; // 1=blue win, -1=red win, 0=draw
  currentState: GameState | null;
  running: boolean;
  speed: number; // speed multiplier per rink
  rinks: number; // number of parallel rinks
  onStatsUpdate: ((s: TrainingStats) => void) | null;
  onStateUpdate: ((s: GameState) => void) | null;
  private timerId: number;
  private disposed: boolean;

  constructor(config?: Partial<TrainingConfig>, existingNetwork?: PolicyNetwork) {
    this.config = { ...DEFAULTS, ...config };
    this.network = existingNetwork || new PolicyNetwork(NETWORK_SIZES);
    this.agent0 = new AIAgent(this.network);
    this.agent1 = new AIAgent(this.network);
    this.stats = {
      episode: 0, blueWins: 0, redWins: 0, draws: 0,
      winRate: 0.5, recentScores: [], avgGoalsPerEp: 0,
    };
    this.recentResults = [];
    this.currentState = null;
    this.running = false;
    this.speed = 5;
    this.rinks = 1;
    this.onStatsUpdate = null;
    this.onStateUpdate = null;
    this.timerId = 0;
    this.disposed = false;
  }

  private runEpisode(): EpisodeResult {
    const possession = this.stats.episode % 2;
    const state = createInitialState(possession);
    this.currentState = state;

    this.agent0.reset();
    this.agent1.reset();

    let prevBlue = 0;
    let prevRed = 0;

    for (let step = 0; step < this.config.maxSteps; step++) {
      if (state.goalMessage) {
        gameStep(state, FIXED_DT);
        continue;
      }

      // Blue agent
      const a0 = this.agent0.act(state, 0);
      if (a0.destination) state.players[0].destination = a0.destination;
      if (a0.steal && state.possession !== 0) {
        attemptSteal(state, 0);
      }

      // Red agent
      const a1 = this.agent1.act(state, 1);
      if (a1.destination) state.players[1].destination = a1.destination;
      if (a1.steal && state.possession !== 1) {
        attemptSteal(state, 1);
      }

      gameStep(state, FIXED_DT);

      // Reward on score change
      if (state.score[0] > prevBlue) {
        this.addReward(this.agent0, 1);
        this.addReward(this.agent1, -1);
        prevBlue = state.score[0];
      }
      if (state.score[1] > prevRed) {
        this.addReward(this.agent1, 1);
        this.addReward(this.agent0, -1);
        prevRed = state.score[1];
      }

      // Small shaping: carrier progress toward goal
      if (!state.goalMessage) {
        const carrier = state.players[state.possession];
        const prog = 0.0005 * (carrier.pos.x / RINK_WIDTH);
        if (state.possession === 0 && this.agent0.trajectory.length > 0)
          this.agent0.trajectory[this.agent0.trajectory.length - 1].reward += prog;
        if (state.possession === 1 && this.agent1.trajectory.length > 0)
          this.agent1.trajectory[this.agent1.trajectory.length - 1].reward += prog;
      }

      // Inactivity penalty: penalize agents that haven't decided in 5+ seconds
      if (this.agent0.stepsSinceLastDecision > INACTIVITY_THRESHOLD && this.agent0.trajectory.length > 0)
        this.agent0.trajectory[this.agent0.trajectory.length - 1].reward += INACTIVITY_PENALTY;
      if (this.agent1.stepsSinceLastDecision > INACTIVITY_THRESHOLD && this.agent1.trajectory.length > 0)
        this.agent1.trajectory[this.agent1.trajectory.length - 1].reward += INACTIVITY_PENALTY;
    }

    // Train both agents
    trainAgent(this.agent0, this.config.gamma, this.config.lr);
    trainAgent(this.agent1, this.config.gamma, this.config.lr);

    return { blue: state.score[0], red: state.score[1] };
  }

  private addReward(agent: AIAgent, r: number): void {
    if (agent.trajectory.length > 0)
      agent.trajectory[agent.trajectory.length - 1].reward += r;
  }

  private updateStats(result: EpisodeResult): void {
    this.stats.episode++;
    if (result.blue > result.red) {
      this.stats.blueWins++;
      this.recentResults.push(1);
    } else if (result.red > result.blue) {
      this.stats.redWins++;
      this.recentResults.push(-1);
    } else {
      this.stats.draws++;
      this.recentResults.push(0);
    }
    if (this.recentResults.length > 100) this.recentResults.shift();

    const wins = this.recentResults.filter(r => r === 1).length;
    this.stats.winRate = wins / this.recentResults.length;

    this.stats.recentScores.push(result);
    if (this.stats.recentScores.length > 200) this.stats.recentScores.shift();

    const totalGoals = this.stats.recentScores.reduce((s, r) => s + r.blue + r.red, 0);
    this.stats.avgGoalsPerEp = totalGoals / this.stats.recentScores.length;

    // Decay exploration
    this.agent0.explorationRate = Math.max(0.02, 0.15 - this.stats.episode * 0.0003);
    this.agent1.explorationRate = this.agent0.explorationRate;
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.disposed = false;
    this.tick();
  }

  stop(): void {
    this.running = false;
    if (this.timerId) clearTimeout(this.timerId);
  }

  dispose(): void {
    this.stop();
    this.disposed = true;
  }

  private tick = (): void => {
    if (!this.running || this.disposed) return;

    const batchSize = Math.max(1, this.speed) * Math.max(1, this.rinks);
    const deadline = performance.now() + 80; // max 80ms per tick to keep UI responsive
    let count = 0;
    while (count < batchSize && performance.now() < deadline) {
      const result = this.runEpisode();
      this.updateStats(result);
      count++;
    }

    if (this.onStatsUpdate) this.onStatsUpdate({ ...this.stats });
    if (this.onStateUpdate && this.currentState) this.onStateUpdate(this.currentState);

    // Use setTimeout(0) instead of RAF for maximum throughput
    this.timerId = window.setTimeout(this.tick, 0);
  };

  getNetwork(): PolicyNetwork {
    return this.network;
  }
}
