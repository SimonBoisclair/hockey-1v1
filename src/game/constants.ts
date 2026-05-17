// Rink dimensions (feet)
export const RINK_WIDTH = 66;
export const RINK_HEIGHT = 29;
export const CORNER_RADIUS = 5;

// Player
export const PLAYER_RADIUS = 2;
export const PUCK_RADIUS = 0.5;

// Zones
export const GOAL_DEPTH = 5;
export const GOAL_WIDTH = 10;
export const BACKCHECK_DEPTH = 16.5;

// Physics
export const MAX_SPEED = 14; // ft/s
export const MAX_FORCE = 28; // ft/s^2
export const DECEL_DISTANCE = 8; // ft
export const ARRIVAL_THRESHOLD = 0.5; // ft
export const MIN_SPEED = 0.3; // ft/s
export const WALL_RESTITUTION = 0.4;

// Steal mechanics
export const STEAL_MAX_DIST = 8; // ft center-to-center
export const STEAL_MIN_DIST = 4; // ft center-to-center
export const STEAL_CHANCE_FAR = 0.2; // 20% at 8ft
export const STEAL_CHANCE_NEAR = 0.5; // 50% at 4ft
export const STEAL_LOCK_DURATION = 1.0; // seconds locked after steal attempt
export const STEAL_DIR_BONUS = 0.25; // max bonus when carrier moving toward stealer
export const STEAL_DIR_PENALTY = 0.15; // max penalty when carrier moving away
export const RESTEAL_DELAY = 1.5; // seconds before victim can attempt to re-steal

// Timing
export const FIXED_DT = 1 / 60;
export const GOAL_DISPLAY_TIME = 2.0; // seconds

// Colors
export const BLUE_COLOR = '#2196F3';
export const BLUE_DARK = '#1565C0';
export const RED_COLOR = '#f44336';
export const RED_DARK = '#c62828';
export const ICE_COLOR = '#e8f4f8';
export const PUCK_COLOR = '#111';
