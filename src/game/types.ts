export interface Vec2 {
  x: number;
  y: number;
}

export interface Player {
  pos: Vec2;
  vel: Vec2;
  destination: Vec2 | null;
  team: 'blue' | 'red';
  mustBackcheck: boolean;
  lockDirection: boolean;
  stealLockTimer: number;
}

export interface GameState {
  players: [Player, Player]; // [blue=0, red=1]
  possession: number; // 0 or 1
  score: [number, number]; // [blue, red]
  paused: boolean;
  goalMessage: string | null;
  goalTimer: number;
  stealMessage: string | null;
  stealMessageTimer: number;
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}
