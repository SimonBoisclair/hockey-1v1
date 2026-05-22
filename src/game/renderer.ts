import { GameState, Vec2 } from './types';
import {
  RINK_WIDTH, RINK_HEIGHT, CORNER_RADIUS, PLAYER_RADIUS, PUCK_RADIUS,
  BACKCHECK_DEPTH,
  BLUE_COLOR, BLUE_DARK, RED_COLOR, RED_DARK, ICE_COLOR, PUCK_COLOR,
  STEAL_MAX_DIST, MIN_SPEED,
} from './constants';
import { getPuckPosition, getGoalZone, getBackcheckZone, v2Length, v2Normalize } from './physics';

function toCanvas(pos: Vec2, scale: number, ox: number, oy: number): [number, number] {
  return [ox + pos.x * scale, oy + pos.y * scale];
}

function drawRoundedRinkPath(ctx: CanvasRenderingContext2D, s: number, ox: number, oy: number): void {
  const r = CORNER_RADIUS * s;
  const w = RINK_WIDTH * s;
  const h = RINK_HEIGHT * s;
  ctx.beginPath();
  ctx.moveTo(ox + r, oy);
  ctx.lineTo(ox + w - r, oy);
  ctx.arcTo(ox + w, oy, ox + w, oy + r, r);
  ctx.lineTo(ox + w, oy + h - r);
  ctx.arcTo(ox + w, oy + h, ox + w - r, oy + h, r);
  ctx.lineTo(ox + r, oy + h);
  ctx.arcTo(ox, oy + h, ox, oy + h - r, r);
  ctx.lineTo(ox, oy + r);
  ctx.arcTo(ox, oy, ox + r, oy, r);
  ctx.closePath();
}

function drawRink(ctx: CanvasRenderingContext2D, s: number, ox: number, oy: number): void {
  // Ice surface
  drawRoundedRinkPath(ctx, s, ox, oy);
  ctx.fillStyle = ICE_COLOR;
  ctx.fill();
  ctx.strokeStyle = '#555';
  ctx.lineWidth = 3;
  ctx.stroke();

  // Center line
  const cx = ox + (RINK_WIDTH / 2) * s;
  ctx.beginPath();
  ctx.moveTo(cx, oy);
  ctx.lineTo(cx, oy + RINK_HEIGHT * s);
  ctx.strokeStyle = '#cc0000';
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 6]);
  ctx.stroke();
  ctx.setLineDash([]);

  // Center circle
  ctx.beginPath();
  ctx.arc(cx, oy + (RINK_HEIGHT / 2) * s, 5 * s, 0, Math.PI * 2);
  ctx.strokeStyle = '#cc0000';
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawZones(ctx: CanvasRenderingContext2D, s: number, ox: number, oy: number): void {
  // Single goal zone (right end)
  const goal = getGoalZone();

  ctx.fillStyle = 'rgba(76, 175, 80, 0.22)';
  ctx.fillRect(ox + goal.x * s, oy + goal.y * s, goal.w * s, goal.h * s);
  ctx.strokeStyle = '#4caf50';
  ctx.lineWidth = 2;
  ctx.strokeRect(ox + goal.x * s, oy + goal.y * s, goal.w * s, goal.h * s);

  // Goal label
  ctx.font = `bold ${Math.max(10, s * 1.4)}px sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#388e3c';
  ctx.fillText('GOAL', ox + (goal.x + goal.w / 2) * s, oy + (goal.y + goal.h / 2) * s);

  // Single backcheck zone (left end)
  const bc = getBackcheckZone();

  ctx.fillStyle = 'rgba(255, 152, 0, 0.08)';
  ctx.fillRect(ox + bc.x * s, oy + bc.y * s, bc.w * s, bc.h * s);

  // Backcheck zone line (dashed)
  ctx.setLineDash([6, 4]);
  ctx.lineWidth = 1;
  ctx.strokeStyle = 'rgba(255, 152, 0, 0.5)';
  ctx.beginPath();
  ctx.moveTo(ox + BACKCHECK_DEPTH * s, oy);
  ctx.lineTo(ox + BACKCHECK_DEPTH * s, oy + RINK_HEIGHT * s);
  ctx.stroke();
  ctx.setLineDash([]);

  // Backcheck label
  ctx.font = `${Math.max(8, s * 0.9)}px sans-serif`;
  ctx.globalAlpha = 0.45;
  ctx.fillStyle = '#ff9800';
  ctx.fillText('BACKCHECK', ox + (BACKCHECK_DEPTH / 2) * s, oy + 2.5 * s);
  ctx.globalAlpha = 1;
}

function drawPlayer(
  ctx: CanvasRenderingContext2D,
  state: GameState,
  playerIdx: number,
  s: number,
  ox: number,
  oy: number,
): void {
  const player = state.players[playerIdx];
  const [px, py] = toCanvas(player.pos, s, ox, oy);
  const r = PLAYER_RADIUS * s;
  const isBlue = player.team === 'blue';
  const hasPuck = state.possession === playerIdx;

  // Puck carrier glow
  if (hasPuck) {
    ctx.beginPath();
    ctx.arc(px, py, r + 3, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 215, 0, 0.35)';
    ctx.fill();
  }

  // Player body
  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.fillStyle = isBlue ? BLUE_COLOR : RED_COLOR;
  ctx.fill();
  ctx.strokeStyle = isBlue ? BLUE_DARK : RED_DARK;
  ctx.lineWidth = 2;
  ctx.stroke();

  // Team letter
  ctx.fillStyle = '#fff';
  ctx.font = `bold ${Math.max(10, r * 0.9)}px sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(isBlue ? 'B' : 'R', px, py + 1);

  // Direction arrow
  const speed = v2Length(player.vel);
  if (speed > MIN_SPEED) {
    const dir = v2Normalize(player.vel);
    const arrowStart = r + 2;
    const arrowEnd = r + 8;
    const ax = px + dir.x * arrowStart;
    const ay = py + dir.y * arrowStart;
    const bx = px + dir.x * arrowEnd;
    const by = py + dir.y * arrowEnd;

    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.strokeStyle = isBlue ? BLUE_DARK : RED_DARK;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Arrowhead
    const angle = Math.atan2(dir.y, dir.x);
    const headLen = 5;
    ctx.beginPath();
    ctx.moveTo(bx, by);
    ctx.lineTo(bx - headLen * Math.cos(angle - 0.5), by - headLen * Math.sin(angle - 0.5));
    ctx.moveTo(bx, by);
    ctx.lineTo(bx - headLen * Math.cos(angle + 0.5), by - headLen * Math.sin(angle + 0.5));
    ctx.stroke();
  }

  // Must backcheck indicator
  if (player.mustBackcheck && hasPuck) {
    ctx.fillStyle = '#ff9800';
    ctx.font = `bold ${Math.max(9, s * 1)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText('BACKCHECK!', px, py - r - 8);
  }
}

function drawPuck(ctx: CanvasRenderingContext2D, state: GameState, s: number, ox: number, oy: number): void {
  const puckPos = getPuckPosition(state);
  const [px, py] = toCanvas(puckPos, s, ox, oy);
  const r = PUCK_RADIUS * s;

  ctx.beginPath();
  ctx.arc(px, py, r, 0, Math.PI * 2);
  ctx.fillStyle = PUCK_COLOR;
  ctx.fill();
  ctx.strokeStyle = '#444';
  ctx.lineWidth = 1;
  ctx.stroke();
}

function drawDestination(
  ctx: CanvasRenderingContext2D,
  state: GameState,
  playerIdx: number,
  s: number,
  ox: number,
  oy: number,
): void {
  const player = state.players[playerIdx];
  if (!player.destination) return;

  const [px, py] = toCanvas(player.pos, s, ox, oy);
  const [dx, dy] = toCanvas(player.destination, s, ox, oy);
  const isBlue = player.team === 'blue';
  const color = isBlue ? BLUE_COLOR : RED_COLOR;

  // Dashed line
  ctx.beginPath();
  ctx.moveTo(px, py);
  ctx.lineTo(dx, dy);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.globalAlpha = 0.5;
  ctx.setLineDash([4, 4]);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;

  // Crosshair
  const cs = 6;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(dx - cs, dy);
  ctx.lineTo(dx + cs, dy);
  ctx.moveTo(dx, dy - cs);
  ctx.lineTo(dx, dy + cs);
  ctx.stroke();

  // Small circle
  ctx.beginPath();
  ctx.arc(dx, dy, 3, 0, Math.PI * 2);
  ctx.stroke();
}

function drawStealRange(
  ctx: CanvasRenderingContext2D,
  state: GameState,
  playerIdx: number,
  s: number,
  ox: number,
  oy: number,
): void {
  const player = state.players[playerIdx];
  if (!player.lockDirection || state.possession === playerIdx) return;

  const [px, py] = toCanvas(player.pos, s, ox, oy);
  const range = STEAL_MAX_DIST * s;

  ctx.beginPath();
  ctx.arc(px, py, range, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(255, 152, 0, 0.4)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 3]);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawHUD(
  ctx: CanvasRenderingContext2D,
  state: GameState,
  cw: number,
  ch: number,
): void {
  // Paused overlay
  if (state.paused) {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
    ctx.fillRect(0, 0, cw, ch);

    ctx.fillStyle = '#fff';
    ctx.font = `bold ${Math.max(24, cw * 0.04)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('PAUSED', cw / 2, ch / 2 - 16);

    ctx.font = `${Math.max(13, cw * 0.02)}px sans-serif`;
    ctx.fillStyle = '#ccc';
    ctx.fillText('Click to set destination, press Space to resume', cw / 2, ch / 2 + 16);
  }

  // Goal message overlay
  if (state.goalMessage) {
    ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
    ctx.fillRect(0, 0, cw, ch);

    ctx.fillStyle = '#ffd700';
    ctx.font = `bold ${Math.max(30, cw * 0.05)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(state.goalMessage, cw / 2, ch / 2);
  }

  // Steal message
  if (state.stealMessage) {
    ctx.fillStyle = state.stealMessage.includes('stolen') ? '#4caf50' : '#ff9800';
    ctx.font = `bold ${Math.max(12, cw * 0.018)}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText(state.stealMessage, cw / 2, 30);
  }
}

// ── Main render ──

export function render(
  ctx: CanvasRenderingContext2D,
  state: GameState,
  canvasWidth: number,
  canvasHeight: number,
  scale: number,
  offsetX: number,
  offsetY: number,
  rotated: boolean = false,
): void {
  // Clear in actual canvas space
  ctx.clearRect(0, 0, canvasWidth, canvasHeight);
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, canvasWidth, canvasHeight);

  // Apply rotation for portrait mode (goal at top)
  if (rotated) {
    ctx.save();
    const virtualW = RINK_WIDTH * scale + offsetX * 2;
    const virtualH = RINK_HEIGHT * scale + offsetY * 2;
    ctx.translate(canvasWidth / 2, canvasHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.translate(-virtualW / 2, -virtualH / 2);
  }

  // Clip to rink shape
  ctx.save();
  drawRoundedRinkPath(ctx, scale, offsetX, offsetY);
  ctx.clip();

  // Draw rink internals (zones draw inside the clip)
  drawRink(ctx, scale, offsetX, offsetY);
  drawZones(ctx, scale, offsetX, offsetY);

  // Draw game objects
  drawDestination(ctx, state, 0, scale, offsetX, offsetY);
  drawDestination(ctx, state, 1, scale, offsetX, offsetY);
  drawStealRange(ctx, state, 0, scale, offsetX, offsetY);
  drawPuck(ctx, state, scale, offsetX, offsetY);
  drawPlayer(ctx, state, 0, scale, offsetX, offsetY);
  drawPlayer(ctx, state, 1, scale, offsetX, offsetY);

  ctx.restore(); // undo clip

  // Rink border (outside clip)
  drawRoundedRinkPath(ctx, scale, offsetX, offsetY);
  ctx.strokeStyle = '#555';
  ctx.lineWidth = 3;
  ctx.stroke();

  if (rotated) {
    ctx.restore(); // undo rotation
  }

  // HUD overlay (always in screen space)
  drawHUD(ctx, state, canvasWidth, canvasHeight);
}
