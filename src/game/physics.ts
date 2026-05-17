import { Vec2, Player, GameState, Rect } from './types';
import {
  RINK_WIDTH, RINK_HEIGHT, PLAYER_RADIUS, PUCK_RADIUS, CORNER_RADIUS,
  MAX_SPEED, MAX_FORCE, DECEL_DISTANCE, ARRIVAL_THRESHOLD, MIN_SPEED,
  WALL_RESTITUTION, BACKCHECK_DEPTH,
  STEAL_MAX_DIST, STEAL_MIN_DIST, STEAL_CHANCE_FAR, STEAL_CHANCE_NEAR,
  STEAL_LOCK_DURATION, STEAL_DIR_BONUS, STEAL_DIR_PENALTY, RESTEAL_DELAY,
  GOAL_DISPLAY_TIME, GOAL_DEPTH, GOAL_WIDTH,
} from './constants';

// ── Vec2 utilities ──

export function v2(x: number, y: number): Vec2 {
  return { x, y };
}

export function v2Add(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x + b.x, y: a.y + b.y };
}

export function v2Sub(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x - b.x, y: a.y - b.y };
}

export function v2Scale(v: Vec2, s: number): Vec2 {
  return { x: v.x * s, y: v.y * s };
}

export function v2Length(v: Vec2): number {
  return Math.sqrt(v.x * v.x + v.y * v.y);
}

export function v2Normalize(v: Vec2): Vec2 {
  const len = v2Length(v);
  if (len < 1e-8) return { x: 0, y: 0 };
  return { x: v.x / len, y: v.y / len };
}

export function v2Distance(a: Vec2, b: Vec2): number {
  return v2Length(v2Sub(a, b));
}

export function v2Dot(a: Vec2, b: Vec2): number {
  return a.x * b.x + a.y * b.y;
}

// ── Zone helpers ──

function getGoalZoneRect(): Rect {
  const goalY = (RINK_HEIGHT - GOAL_WIDTH) / 2;
  return { x: RINK_WIDTH - GOAL_DEPTH, y: goalY, w: GOAL_DEPTH, h: GOAL_WIDTH };
}

function getBackcheckZoneRect(): Rect {
  return { x: 0, y: 0, w: BACKCHECK_DEPTH, h: RINK_HEIGHT };
}

function pointInRect(p: Vec2, r: Rect): boolean {
  return p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h;
}

// ── State creation ──

export function createInitialState(possession: number): GameState {
  // Carrier always starts on the left, defender on the right
  const carrierPos = v2(12, RINK_HEIGHT / 2);
  const defenderPos = v2(40, RINK_HEIGHT / 2);

  const bluePos = possession === 0 ? carrierPos : defenderPos;
  const redPos = possession === 1 ? carrierPos : defenderPos;

  return {
    players: [
      {
        pos: v2(bluePos.x, bluePos.y),
        vel: v2(0, 0),
        destination: null,
        team: 'blue',
        mustBackcheck: false,
        lockDirection: false,
        stealLockTimer: 0,
      },
      {
        pos: v2(redPos.x, redPos.y),
        vel: v2(0, 0),
        destination: null,
        team: 'red',
        mustBackcheck: false,
        lockDirection: false,
        stealLockTimer: 0,
      },
    ],
    possession,
    score: [0, 0],
    paused: false,
    goalMessage: null,
    goalTimer: 0,
    stealMessage: null,
    stealMessageTimer: 0,
  };
}

export function resetAfterGoal(state: GameState, newPossession: number): void {
  // Carrier always starts on the left, defender on the right
  const carrierPos = v2(12, RINK_HEIGHT / 2);
  const defenderPos = v2(40, RINK_HEIGHT / 2);

  state.players[0].pos = newPossession === 0 ? v2(carrierPos.x, carrierPos.y) : v2(defenderPos.x, defenderPos.y);
  state.players[1].pos = newPossession === 1 ? v2(carrierPos.x, carrierPos.y) : v2(defenderPos.x, defenderPos.y);
  for (const p of state.players) {
    p.vel = v2(0, 0);
    p.destination = null;
    p.mustBackcheck = false;
    p.lockDirection = false;
    p.stealLockTimer = 0;
  }
  state.possession = newPossession;
  state.goalMessage = null;
  state.goalTimer = 0;
}

// ── Player physics ──

function stepPlayer(player: Player, dt: number): void {
  if (!player.destination) return;
  if (player.lockDirection) {
    // Move straight, no steering
    player.pos = v2Add(player.pos, v2Scale(player.vel, dt));
    return;
  }

  const toTarget = v2Sub(player.destination, player.pos);
  const dist = v2Length(toTarget);

  // Arrived?
  if (dist < ARRIVAL_THRESHOLD) {
    const speed = v2Length(player.vel);
    if (speed < MIN_SPEED) {
      player.vel = v2(0, 0);
      player.destination = null;
      return;
    }
  }

  // Desired speed based on distance (deceleration near target)
  let desiredSpeed = MAX_SPEED;
  if (dist < DECEL_DISTANCE) {
    desiredSpeed = MAX_SPEED * Math.max(dist / DECEL_DISTANCE, 0.05);
  }

  // Desired velocity
  const desiredDir = v2Normalize(toTarget);
  const desiredVel = v2Scale(desiredDir, desiredSpeed);

  // Steering force
  let steering = v2Sub(desiredVel, player.vel);
  const steeringMag = v2Length(steering);
  if (steeringMag > MAX_FORCE) {
    steering = v2Scale(v2Normalize(steering), MAX_FORCE);
  }

  // Apply steering
  player.vel = v2Add(player.vel, v2Scale(steering, dt));

  // Clamp speed
  const speed = v2Length(player.vel);
  if (speed > MAX_SPEED) {
    player.vel = v2Scale(v2Normalize(player.vel), MAX_SPEED);
  }

  // Update position
  player.pos = v2Add(player.pos, v2Scale(player.vel, dt));
}

// ── Wall collisions ──

function handleWallCollision(player: Player): void {
  const r = PLAYER_RADIUS;

  // Straight walls
  if (player.pos.x - r < 0) {
    player.pos.x = r;
    player.vel.x = Math.abs(player.vel.x) * WALL_RESTITUTION;
  }
  if (player.pos.x + r > RINK_WIDTH) {
    player.pos.x = RINK_WIDTH - r;
    player.vel.x = -Math.abs(player.vel.x) * WALL_RESTITUTION;
  }
  if (player.pos.y - r < 0) {
    player.pos.y = r;
    player.vel.y = Math.abs(player.vel.y) * WALL_RESTITUTION;
  }
  if (player.pos.y + r > RINK_HEIGHT) {
    player.pos.y = RINK_HEIGHT - r;
    player.vel.y = -Math.abs(player.vel.y) * WALL_RESTITUTION;
  }

  // Rounded corners
  const corners = [
    v2(CORNER_RADIUS, CORNER_RADIUS),
    v2(RINK_WIDTH - CORNER_RADIUS, CORNER_RADIUS),
    v2(CORNER_RADIUS, RINK_HEIGHT - CORNER_RADIUS),
    v2(RINK_WIDTH - CORNER_RADIUS, RINK_HEIGHT - CORNER_RADIUS),
  ];
  const cornerChecks = [
    (p: Vec2) => p.x < CORNER_RADIUS && p.y < CORNER_RADIUS,
    (p: Vec2) => p.x > RINK_WIDTH - CORNER_RADIUS && p.y < CORNER_RADIUS,
    (p: Vec2) => p.x < CORNER_RADIUS && p.y > RINK_HEIGHT - CORNER_RADIUS,
    (p: Vec2) => p.x > RINK_WIDTH - CORNER_RADIUS && p.y > RINK_HEIGHT - CORNER_RADIUS,
  ];

  for (let i = 0; i < 4; i++) {
    if (cornerChecks[i](player.pos)) {
      const center = corners[i];
      const diff = v2Sub(player.pos, center);
      const dist = v2Length(diff);
      const maxDist = CORNER_RADIUS - r;
      if (dist > maxDist && dist > 0) {
        const dir = v2Normalize(diff);
        player.pos = v2Add(center, v2Scale(dir, maxDist));
        // Reflect velocity along corner normal
        const velNormal = v2Dot(player.vel, dir);
        if (velNormal > 0) {
          const normalComp = v2Scale(dir, velNormal);
          player.vel = v2Sub(player.vel, v2Scale(normalComp, 1 + WALL_RESTITUTION));
        }
      }
    }
  }
}

// ── Player-player collision ──

function handlePlayerCollision(p0: Player, p1: Player): void {
  const diff = v2Sub(p0.pos, p1.pos);
  const dist = v2Length(diff);
  const minDist = PLAYER_RADIUS * 2;

  if (dist < minDist && dist > 0.01) {
    const dir = v2Normalize(diff);
    const overlap = minDist - dist;

    // Push apart
    p0.pos = v2Add(p0.pos, v2Scale(dir, overlap / 2));
    p1.pos = v2Sub(p1.pos, v2Scale(dir, overlap / 2));

    // Elastic bounce
    const relVel = v2Sub(p0.vel, p1.vel);
    const velAlongNormal = v2Dot(relVel, dir);
    if (velAlongNormal < 0) {
      const impulse = v2Scale(dir, velAlongNormal);
      p0.vel = v2Sub(p0.vel, impulse);
      p1.vel = v2Add(p1.vel, impulse);
    }
  }
}

// ── Scoring ──

function checkGoalScored(state: GameState): 'blue' | 'red' | null {
  const carrier = state.players[state.possession];
  if (carrier.mustBackcheck) return null;

  const zone = getGoalZoneRect();
  if (pointInRect(carrier.pos, zone)) {
    return carrier.team;
  }
  return null;
}

// ── Backcheck ──

function updateBackcheck(player: Player): void {
  if (!player.mustBackcheck) return;
  const zone = getBackcheckZoneRect();
  if (pointInRect(player.pos, zone)) {
    player.mustBackcheck = false;
  }
}

// ── Steal ──

function enforceGoalZoneRestriction(player: Player): void {
  const goalZoneX = RINK_WIDTH - GOAL_DEPTH;
  const goalY = (RINK_HEIGHT - GOAL_WIDTH) / 2;
  // Only restrict if player is within the goal zone's vertical range
  if (player.pos.y >= goalY && player.pos.y <= goalY + GOAL_WIDTH) {
    // Non-carrier center cannot go past goalZoneX (half body allowed in)
    if (player.pos.x > goalZoneX) {
      player.pos.x = goalZoneX;
      if (player.vel.x > 0) player.vel.x = 0;
    }
  }
}

export function attemptSteal(state: GameState, stealerIndex: number): { success: boolean; message: string } {
  if (state.possession === stealerIndex) {
    return { success: false, message: '' };
  }

  const stealer = state.players[stealerIndex];

  // Check if still locked from a previous steal attempt
  if (stealer.stealLockTimer > 0) {
    return { success: false, message: 'Locked! Wait...' };
  }

  const carrier = state.players[state.possession];
  const dist = v2Distance(stealer.pos, carrier.pos);

  // Lock direction for 1 second regardless of outcome
  stealer.lockDirection = true;
  stealer.stealLockTimer = STEAL_LOCK_DURATION;

  if (dist > STEAL_MAX_DIST) {
    return { success: false, message: 'Too far to steal!' };
  }

  // Base chance from distance
  let chance: number;
  if (dist <= STEAL_MIN_DIST) {
    chance = STEAL_CHANCE_NEAR;
  } else {
    const t = (dist - STEAL_MIN_DIST) / (STEAL_MAX_DIST - STEAL_MIN_DIST);
    chance = STEAL_CHANCE_NEAR + t * (STEAL_CHANCE_FAR - STEAL_CHANCE_NEAR);
  }

  // Directional modifier: carrier velocity relative to stealer
  const carrierSpeed = v2Length(carrier.vel);
  if (carrierSpeed > MIN_SPEED) {
    const toStealer = v2Normalize(v2Sub(stealer.pos, carrier.pos));
    const carrierDir = v2Normalize(carrier.vel);
    const dot = v2Dot(carrierDir, toStealer); // +1 = toward stealer, -1 = away
    if (dot > 0) {
      chance += dot * STEAL_DIR_BONUS;
    } else {
      chance += dot * STEAL_DIR_PENALTY; // dot is negative, so this subtracts
    }
    chance = Math.max(0.05, Math.min(0.85, chance));
  }

  const roll = Math.random();
  const pct = Math.round(chance * 100);

  if (roll < chance) {
    const victimIdx = state.possession;
    state.possession = stealerIndex;
    stealer.mustBackcheck = true;
    // Prevent the victim from immediately re-stealing
    state.players[victimIdx].stealLockTimer = RESTEAL_DELAY;
    return { success: true, message: `Puck stolen! (${pct}% chance)` };
  }

  return { success: false, message: `Steal failed (${pct}% chance)` };
}

// ── Main game step ──

export function gameStep(state: GameState, dt: number): void {
  // Handle goal display timer
  if (state.goalMessage) {
    state.goalTimer -= dt;
    if (state.goalTimer <= 0) {
      const scoredByBlue = state.goalMessage.includes('BLUE');
      resetAfterGoal(state, scoredByBlue ? 1 : 0);
    }
    return; // Freeze during goal celebration
  }

  // Handle steal message timer
  if (state.stealMessage) {
    state.stealMessageTimer -= dt;
    if (state.stealMessageTimer <= 0) {
      state.stealMessage = null;
    }
  }

  // Tick steal lock timers and auto-unlock
  for (const player of state.players) {
    if (player.stealLockTimer > 0) {
      player.stealLockTimer -= dt;
      if (player.stealLockTimer <= 0) {
        player.stealLockTimer = 0;
        player.lockDirection = false;
      }
    }
  }

  // Step each player
  for (const player of state.players) {
    stepPlayer(player, dt);
  }

  // Wall collisions
  for (const player of state.players) {
    handleWallCollision(player);
  }

  // Player-player collision
  handlePlayerCollision(state.players[0], state.players[1]);

  // Non-carrier cannot enter more than half body into goal zone
  const nonCarrierIdx = 1 - state.possession;
  enforceGoalZoneRestriction(state.players[nonCarrierIdx]);

  // Update backcheck status
  const carrier = state.players[state.possession];
  updateBackcheck(carrier);

  // Check for goal
  const scored = checkGoalScored(state);
  if (scored) {
    const idx = scored === 'blue' ? 0 : 1;
    state.score[idx]++;
    state.goalMessage = `GOAL! ${scored.toUpperCase()} scores!`;
    state.goalTimer = GOAL_DISPLAY_TIME;
  }
}

// ── Puck position (derived from carrier) ──

export function getPuckPosition(state: GameState): Vec2 {
  const carrier = state.players[state.possession];
  const speed = v2Length(carrier.vel);
  let dir: Vec2;
  if (speed > MIN_SPEED) {
    dir = v2Normalize(carrier.vel);
  } else {
    dir = v2(1, 0); // puck faces toward goal (right)
  }
  const offset = PLAYER_RADIUS + PUCK_RADIUS + 0.3;
  return v2Add(carrier.pos, v2Scale(dir, offset));
}

// ── Goal and backcheck zone getters (for renderer) ──

export function getGoalZone(): Rect {
  return getGoalZoneRect();
}

export function getBackcheckZone(): Rect {
  return getBackcheckZoneRect();
}
