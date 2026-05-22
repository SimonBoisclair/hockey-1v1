import { useRef, useEffect, useState, useCallback } from 'react';
import { GameState } from './game/types';
import {
  RINK_WIDTH, RINK_HEIGHT, FIXED_DT, PLAYER_RADIUS,
} from './game/constants';
import { createInitialState, gameStep, attemptSteal } from './game/physics';
import { render } from './game/renderer';
import { AIAgent, PolicyNetwork } from './game/ai';
import { Trainer } from './game/training';
import { listModels, getModel, deleteModel, SavedModel, startGpuTraining, stopGpuTraining, getTrainingStatus, TrainingStatus } from './api';
import './App.css';

type GameMode = 'practice' | 'training' | 'play-ai' | 'admin';

const AI_LEVELS = [
  { level: 1, name: 'Level 1', label: 'Beginner', target: 1_000_000 },
  { level: 2, name: 'Level 2', label: 'Intermediate', target: 5_000_000 },
  { level: 3, name: 'Level 3', label: 'Advanced', target: 10_000_000 },
] as const;

function modelNameForLevel(level: number): string {
  return `AI-Level-${level}`;
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<GameState>(createInitialState(0));
  const rafRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);
  const accumulatorRef = useRef<number>(0);
  const scaleRef = useRef<number>(1);
  const offsetRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const isPortraitRef = useRef(false);

  const [isPortrait, setIsPortrait] = useState(false);

  const trainerRef = useRef<Trainer | null>(null);
  const aiAgentRef = useRef<AIAgent | null>(null);
  const networkRef = useRef<PolicyNetwork | null>(null);

  const [mode, setMode] = useState<GameMode>('practice');
  const [score, setScore] = useState<[number, number]>([0, 0]);
  const [paused, setPaused] = useState(false);
  const [trainingRunning, setTrainingRunning] = useState(false);
  const [savedModels, setSavedModels] = useState<SavedModel[]>([]);
  const [loadingModel, setLoadingModel] = useState(false);
  const [statusMsg, setStatusMsg] = useState('');
  const [trainingLevel, setTrainingLevel] = useState<number | null>(null);
  const [levelModels, setLevelModels] = useState<Record<number, SavedModel | null>>({});
  const [playLevel, setPlayLevel] = useState<number | null>(null);

  const [gpuStatus, setGpuStatus] = useState<TrainingStatus | null>(null);
  const gpuPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const PADDING = 10;

  const resizeCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const vh = window.innerHeight;
    const vw = window.innerWidth;
    const dpr = window.devicePixelRatio || 1;
    const portraitMobile = vw <= 900 && vw < vh;
    isPortraitRef.current = portraitMobile;
    setIsPortrait(portraitMobile);

    if (portraitMobile) {
      const sidebarW = 55;
      const bottomBtnH = 52;
      const gapSpace = 6;
      const displayW = vw - sidebarW - gapSpace;
      const displayH = vh - bottomBtnH - gapSpace;
      const scaleByWidth = (displayW - PADDING * 2) / RINK_HEIGHT;
      const scaleByHeight = (displayH - PADDING * 2) / RINK_WIDTH;
      const s = Math.min(scaleByWidth, scaleByHeight);
      const internalW = RINK_HEIGHT * s + PADDING * 2;
      const internalH = RINK_WIDTH * s + PADDING * 2;
      canvas.width = internalW * dpr;
      canvas.height = internalH * dpr;
      canvas.style.width = internalW + 'px';
      canvas.style.height = internalH + 'px';
      const ctx = canvas.getContext('2d');
      if (ctx) ctx.scale(dpr, dpr);
      scaleRef.current = s;
      offsetRef.current = { x: PADDING, y: PADDING };
      return;
    }

    const isMobile = vw <= 900;
    const reservedHeight = isMobile ? 80 : 160;
    const availableHeight = vh - reservedHeight;
    const availableWidth = Math.min(container.clientWidth, 1100) - PADDING * 2;
    const scaleByWidth = availableWidth / RINK_WIDTH;
    const scaleByHeight = (availableHeight - PADDING * 2) / RINK_HEIGHT;
    const s = Math.min(scaleByWidth, scaleByHeight);
    const cw = RINK_WIDTH * s + PADDING * 2;
    const ch = RINK_HEIGHT * s + PADDING * 2;
    canvas.width = cw * dpr;
    canvas.height = ch * dpr;
    canvas.style.width = cw + 'px';
    canvas.style.height = ch + 'px';
    const ctx = canvas.getContext('2d');
    if (ctx) ctx.scale(dpr, dpr);
    scaleRef.current = s;
    offsetRef.current = { x: PADDING, y: PADDING };
  }, []);

  const screenToGame = useCallback((clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const s = scaleRef.current;
    const ox = offsetRef.current.x;
    const oy = offsetRef.current.y;

    if (isPortraitRef.current) {
      const dpr = window.devicePixelRatio || 1;
      const internalW = canvas.width / dpr;
      const internalH = canvas.height / dpr;
      const ix = (clientX - rect.left) * (internalW / rect.width);
      const iy = (clientY - rect.top) * (internalH / rect.height);
      const gameX = (internalH - iy - ox) / s;
      const gameY = (ix - oy) / s;
      return {
        x: Math.max(PLAYER_RADIUS, Math.min(RINK_WIDTH - PLAYER_RADIUS, gameX)),
        y: Math.max(PLAYER_RADIUS, Math.min(RINK_HEIGHT - PLAYER_RADIUS, gameY)),
      };
    }

    const x = (clientX - rect.left - ox) / s;
    const y = (clientY - rect.top - oy) / s;
    return {
      x: Math.max(PLAYER_RADIUS, Math.min(RINK_WIDTH - PLAYER_RADIUS, x)),
      y: Math.max(PLAYER_RADIUS, Math.min(RINK_HEIGHT - PLAYER_RADIUS, y)),
    };
  }, []);

  const setBlueDestination = useCallback((clientX: number, clientY: number) => {
    const pos = screenToGame(clientX, clientY);
    if (!pos) return;
    stateRef.current.players[0].destination = pos;
  }, [screenToGame]);

  const togglePause = useCallback(() => {
    const state = stateRef.current;
    state.paused = !state.paused;
    setPaused(state.paused);
  }, []);

  const releasePause = useCallback(() => {
    stateRef.current.paused = false;
    setPaused(false);
  }, []);

  const doSteal = useCallback(() => {
    const state = stateRef.current;
    const result = attemptSteal(state, 0);
    if (result.message) {
      state.stealMessage = result.message;
      state.stealMessageTimer = 1.5;
    }
  }, []);

  const fetchModels = useCallback(async () => {
    try {
      const models = await listModels();
      setSavedModels(models);
      const lvlMap: Record<number, SavedModel | null> = {};
      for (const lvl of AI_LEVELS) {
        lvlMap[lvl.level] = models.find(m => m.name === modelNameForLevel(lvl.level)) || null;
      }
      setLevelModels(lvlMap);
    } catch {
      console.error('Failed to fetch models');
    }
  }, []);

  const handleLoad = useCallback(async (id: number) => {
    setLoadingModel(true);
    setStatusMsg('');
    try {
      const model = await getModel(id);
      const net = PolicyNetwork.deserialize(model.weights!);
      networkRef.current = net;
      if (trainerRef.current) {
        trainerRef.current = new Trainer(undefined, net);
      }
      setStatusMsg(`Loaded "${model.name}"`);
    } catch {
      setStatusMsg('Failed to load model');
    }
    setLoadingModel(false);
  }, []);

  const handleDelete = useCallback(async (id: number, name: string) => {
    if (!confirm(`Delete model "${name}"?`)) return;
    try {
      await deleteModel(id);
      setStatusMsg(`Deleted "${name}"`);
      await fetchModels();
    } catch {
      setStatusMsg('Failed to delete model');
    }
  }, [fetchModels]);

  const startGpuPoll = useCallback(() => {
    if (gpuPollRef.current) clearInterval(gpuPollRef.current);
    gpuPollRef.current = setInterval(async () => {
      try {
        const status = await getTrainingStatus();
        setGpuStatus(status);
        if (status.status === 'completed' || status.status === 'idle' || status.status === 'stopped' || status.status?.startsWith('error')) {
          if (gpuPollRef.current) clearInterval(gpuPollRef.current);
          gpuPollRef.current = null;
          setTrainingRunning(false);
          if (status.status === 'completed') {
            setStatusMsg(`Level ${status.level || trainingLevel || '?'} training complete!`);
            fetchModels();
          } else if (status.status?.startsWith('error')) {
            setStatusMsg(`Training error: ${status.status}`);
          }
        }
      } catch {
        // network error, keep polling
      }
    }, 5000);
  }, [fetchModels, trainingLevel]);

  const startLevelTraining = useCallback(async (level: number) => {
    setTrainingLevel(level);
    setStatusMsg(`Starting Level ${level} GPU training...`);
    const lvl = AI_LEVELS.find(l => l.level === level);
    if (!lvl) return;

    const prevLevel = level > 1 ? levelModels[level - 1] : null;
    const currentModel = levelModels[level];
    const loadModelId = currentModel?.id?.toString() || prevLevel?.id?.toString();

    try {
      await startGpuTraining({
        model_name: modelNameForLevel(level),
        episodes: lvl.target,
        save_interval: 50000,
        level,
        load_model_id: loadModelId,
        compat_mode: true,
      });
      setTrainingRunning(true);
      setStatusMsg(`Level ${level} GPU training started! You can leave this page.`);
      startGpuPoll();
    } catch (err) {
      setStatusMsg(`Failed to start training: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [levelModels, startGpuPoll]);

  const handleStopGpuTraining = useCallback(async () => {
    try {
      await stopGpuTraining();
      setTrainingRunning(false);
      setStatusMsg('Training stopped');
      if (gpuPollRef.current) clearInterval(gpuPollRef.current);
      gpuPollRef.current = null;
    } catch {
      setStatusMsg('Failed to stop training');
    }
  }, []);

  const selectPlayLevel = useCallback(async (level: number) => {
    const model = levelModels[level];
    if (!model) return;
    setLoadingModel(true);
    setPlayLevel(level);
    try {
      const fullModel = await getModel(model.id);
      const net = PolicyNetwork.deserialize(fullModel.weights!);
      networkRef.current = net;
      aiAgentRef.current = new AIAgent(net);
      stateRef.current = createInitialState(0);
      setScore([0, 0]);
      setPaused(false);
      setMode('play-ai');
    } catch {
      setStatusMsg('Failed to load model');
    }
    setLoadingModel(false);
  }, [levelModels]);

  // Mode switching
  const switchMode = useCallback((newMode: GameMode) => {
    if (trainerRef.current) trainerRef.current.stop();
    cancelAnimationFrame(rafRef.current);
    lastTimeRef.current = 0;
    accumulatorRef.current = 0;
    if (newMode === 'practice') {
      stateRef.current = createInitialState(0);
      setScore([0, 0]);
      setPaused(false);
    }
    if (newMode === 'play-ai') {
      setPlayLevel(null);
      fetchModels();
    }
    if (newMode === 'training') {
      // Check for ongoing GPU training on server
      getTrainingStatus().then(status => {
        setGpuStatus(status);
        if (status.status === 'training' || status.status === 'starting') {
          setTrainingRunning(true);
          setTrainingLevel(status.level || null);
          startGpuPoll();
        }
      }).catch(() => {});
    }
    if (newMode === 'admin') {
      fetchModels();
    }
    setMode(newMode);
    setTrainingRunning(false);
    setStatusMsg('');
  }, [fetchModels, startGpuPoll]);

  // Check for ongoing GPU training on mount
  useEffect(() => {
    getTrainingStatus().then(status => {
      setGpuStatus(status);
      if (status.status === 'training' || status.status === 'starting') {
        setTrainingRunning(true);
        setTrainingLevel(status.level || null);
        startGpuPoll();
      }
    }).catch(() => {});
    return () => {
      if (gpuPollRef.current) clearInterval(gpuPollRef.current);
    };
  }, [startGpuPoll]);

  // Fetch level models on mount
  useEffect(() => { fetchModels(); }, [fetchModels]);

  // Game loop (practice + play-ai)
  useEffect(() => {
    if (mode === 'training' || mode === 'admin') return;
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    const loop = (time: number) => {
      const state = stateRef.current;
      if (lastTimeRef.current === 0) lastTimeRef.current = time;
      const rawDt = (time - lastTimeRef.current) / 1000;
      lastTimeRef.current = time;
      const dt = Math.min(rawDt, 0.1);
      if (!state.paused) {
        accumulatorRef.current += dt;
        while (accumulatorRef.current >= FIXED_DT) {
          if (mode === 'play-ai' && aiAgentRef.current && !state.goalMessage) {
            const aiAction = aiAgentRef.current.act(state, 1);
            if (aiAction.destination) state.players[1].destination = aiAction.destination;
            if (aiAction.steal && state.possession !== 1) {
              const stealResult = attemptSteal(state, 1);
              if (stealResult.message) {
                state.stealMessage = stealResult.message;
                state.stealMessageTimer = 1.5;
              }
            }
          }
          gameStep(state, FIXED_DT);
          accumulatorRef.current -= FIXED_DT;
        }
      }
      setScore(prev => {
        const cs = state.score;
        if (prev[0] !== cs[0] || prev[1] !== cs[1]) return [cs[0], cs[1]];
        return prev;
      });
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.save();
          const dpr = window.devicePixelRatio || 1;
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          render(ctx, state, canvas.width / dpr, canvas.height / dpr, scaleRef.current, offsetRef.current.x, offsetRef.current.y, isPortraitRef.current);
          ctx.restore();
        }
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => { cancelAnimationFrame(rafRef.current); window.removeEventListener('resize', resizeCanvas); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  // Training mode render loop
  useEffect(() => {
    if (mode !== 'training') return;
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    const renderLoop = () => {
      const trainer = trainerRef.current;
      if (trainer && trainer.currentState) {
        const canvas = canvasRef.current;
        if (canvas) {
          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.save();
            const dpr = window.devicePixelRatio || 1;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            render(ctx, trainer.currentState, canvas.width / dpr, canvas.height / dpr, scaleRef.current, offsetRef.current.x, offsetRef.current.y, isPortraitRef.current);
            ctx.restore();
          }
        }
      }
      rafRef.current = requestAnimationFrame(renderLoop);
    };
    rafRef.current = requestAnimationFrame(renderLoop);
    return () => { cancelAnimationFrame(rafRef.current); window.removeEventListener('resize', resizeCanvas); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  // Keyboard events
  useEffect(() => {
    if (mode === 'training' || mode === 'admin') return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space') { e.preventDefault(); togglePause(); }
      if (e.key === 'c' || e.key === 'C') { e.preventDefault(); doSteal(); }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => { window.removeEventListener('keydown', handleKeyDown); };
  }, [mode, togglePause, doSteal]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (mode === 'training' || mode === 'admin') return;
    setBlueDestination(e.clientX, e.clientY);
  };

  const handleCanvasTouch = (e: React.TouchEvent<HTMLCanvasElement>) => {
    if (mode === 'training' || mode === 'admin') return;
    e.preventDefault();
    const touch = e.changedTouches[0];
    if (touch) setBlueDestination(touch.clientX, touch.clientY);
  };

  const isPlayMode = mode === 'practice' || (mode === 'play-ai' && !!playLevel);
  const isGameMode = mode === 'practice' || (mode === 'play-ai' && !!playLevel) || mode === 'training';
  return (
    <div className={'app' + (isPortrait ? ' portrait' : '')}>
      {!isPortrait && (
        <div className="landscape-prompt">
          <div className="landscape-prompt-content">
            <span className="rotate-icon">&#x1F504;</span>
            <p>Rotate your phone to landscape</p>
          </div>
        </div>
      )}

      {!isPortrait && (
        <>
          <div className="mode-bar">
            <button className={'mode-btn' + (mode === 'practice' ? ' active' : '')} onClick={() => switchMode('practice')}>Practice</button>
            <button className={'mode-btn' + (mode === 'training' ? ' active' : '')} onClick={() => switchMode('training')}>Train AI</button>
            <button className={'mode-btn' + (mode === 'play-ai' ? ' active' : '')} onClick={() => switchMode('play-ai')}>Play vs AI</button>
            <button className={'mode-btn' + (mode === 'admin' ? ' active' : '')} onClick={() => switchMode('admin')}>Models</button>
          </div>

          {isPlayMode && (
            <div className="scoreboard">
              <span className="team-blue">BLUE</span>
              <span className="score-value">{score[0]}</span>
              <span className="score-dash">&mdash;</span>
              <span className="score-value">{score[1]}</span>
              <span className="team-red">RED</span>
            </div>
          )}


        </>
      )}

      {isPortrait ? (
        (() => {
          const m = mode as string;
          const showCanvas = isGameMode && mode !== 'training';
          return showCanvas ? (
            <div className="game-area">
              <div className="canvas-container" ref={containerRef}>
                <canvas ref={canvasRef} onClick={handleCanvasClick} onTouchStart={handleCanvasTouch} />
              </div>
              <div className="portrait-sidebar">
                <button className={'sidebar-btn' + (m === 'practice' ? ' active' : '')} onClick={() => switchMode('practice')}>Practice</button>
                <button className={'sidebar-btn' + (m === 'training' ? ' active' : '')} onClick={() => switchMode('training')}>Train</button>
                <button className={'sidebar-btn' + (m === 'play-ai' ? ' active' : '')} onClick={() => switchMode('play-ai')}>Play AI</button>
                <button className={'sidebar-btn' + (m === 'admin' ? ' active' : '')} onClick={() => switchMode('admin')}>Models</button>
                {isPlayMode && (
                  <div className="sidebar-score">
                    <span className="team-blue">{score[0]}</span>
                    <span className="score-dash">&ndash;</span>
                    <span className="team-red">{score[1]}</span>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="portrait-nav-bar">
              <button className={'nav-btn' + (m === 'practice' ? ' active' : '')} onClick={() => switchMode('practice')}>Practice</button>
              <button className={'nav-btn' + (m === 'training' ? ' active' : '')} onClick={() => switchMode('training')}>Train</button>
              <button className={'nav-btn' + (m === 'play-ai' ? ' active' : '')} onClick={() => switchMode('play-ai')}>Play AI</button>
              <button className={'nav-btn' + (m === 'admin' ? ' active' : '')} onClick={() => switchMode('admin')}>Models</button>
            </div>
          );
        })()
      ) : (
        isGameMode && (
          <div className="game-area">
            <div className="canvas-container" ref={containerRef}>
              <canvas ref={canvasRef} onClick={handleCanvasClick} onTouchStart={handleCanvasTouch} />
            </div>
          </div>
        )
      )}
      {isPortrait && isPlayMode && (
        <div className="portrait-controls">
          <button
            className={'bottom-btn pause-bottom' + (paused ? ' active' : '')}
            onPointerDown={(e) => {
              e.preventDefault();
              (e.target as HTMLElement).setPointerCapture(e.pointerId);
              stateRef.current.paused = true;
              setPaused(true);
            }}
            onPointerUp={(e) => { e.preventDefault(); releasePause(); }}
            onPointerCancel={(e) => { e.preventDefault(); releasePause(); }}
            style={{ touchAction: 'none' }}
          >
            <span className="bottom-btn-icon">{paused ? '\u25B6' : '\u23F8'}</span>
            <span className="bottom-btn-label">PAUSE</span>
          </button>
          <button
            className="bottom-btn steal-bottom"
            onTouchStart={(e) => { e.preventDefault(); doSteal(); }}
            onMouseDown={(e) => { e.preventDefault(); doSteal(); }}
          >
            <span className="bottom-btn-icon">{'\u26A1'}</span>
            <span className="bottom-btn-label">STEAL</span>
          </button>
        </div>
      )}

      {isPlayMode && !isPortrait && (
        <div className="mobile-controls">
          <button className={'ctrl-btn pause-btn' + (paused ? ' active' : '')} onMouseDown={(e) => { e.preventDefault(); togglePause(); }} onTouchStart={(e) => { e.preventDefault(); togglePause(); }}>
            {paused ? '\u25B6 RESUME' : '\u23F8 PAUSE'}
          </button>
          <button className="ctrl-btn steal-btn" onMouseDown={(e) => { e.preventDefault(); doSteal(); }} onTouchStart={(e) => { e.preventDefault(); doSteal(); }}>
            {'\u26A1'} STEAL
          </button>
        </div>
      )}

      {mode === 'training' && (
        <div className="training-levels">
          <h3>AI Training Levels (GPU)</h3>
          {statusMsg && <div className="status-msg">{statusMsg}</div>}
          {gpuStatus && gpuStatus.status !== 'idle' && (
            <div className="gpu-status-banner">
              <span className="gpu-status-label">GPU: {gpuStatus.status}</span>
              {gpuStatus.eps_per_sec ? <span>{Math.round(gpuStatus.eps_per_sec)} ep/s</span> : null}
              {gpuStatus.cost_per_hr ? <span>${gpuStatus.cost_per_hr}/hr</span> : null}
            </div>
          )}
          {AI_LEVELS.map(lvl => {
            const model = levelModels[lvl.level];
            const isGpuActive = trainingRunning && (gpuStatus?.level === lvl.level || trainingLevel === lvl.level);
            const episodes = isGpuActive && gpuStatus?.episode ? gpuStatus.episode : (model?.episodes || 0);
            const progress = Math.min(1, episodes / lvl.target);
            const isComplete = episodes >= lvl.target;
            const prevComplete = lvl.level === 1 || (levelModels[lvl.level - 1]?.episodes || 0) >= AI_LEVELS[lvl.level - 2]?.target;
            return (
              <div key={lvl.level} className={'level-card' + (isComplete ? ' complete' : '') + (isGpuActive ? ' active' : '')}>
                <div className="level-header">
                  <span className="level-name">{lvl.name}</span>
                  <span className="level-label">{lvl.label}</span>
                </div>
                <div className="level-progress-bar">
                  <div className="level-progress-fill" style={{ width: `${progress * 100}%` }} />
                </div>
                <div className="level-info">
                  <span>{episodes >= 1000 ? `${(episodes / 1000).toFixed(0)}K` : episodes.toLocaleString()} / {(lvl.target / 1_000_000).toFixed(0)}M episodes</span>
                  {isComplete && <span className="level-done">Done</span>}
                  {isGpuActive && <span className="level-running">Training on GPU...</span>}
                </div>
                {isGpuActive && gpuStatus && (
                  <div className="level-live-stats">
                    <span>Blue W: {gpuStatus.blue_wins || 0}</span>
                    <span>Red W: {gpuStatus.red_wins || 0}</span>
                    <span>Draws: {gpuStatus.draws || 0}</span>
                    {gpuStatus.eps_per_sec ? <span>{Math.round(gpuStatus.eps_per_sec)} ep/s</span> : null}
                  </div>
                )}
                {!isComplete && prevComplete && !trainingRunning && (
                  <button className="ctrl-btn start-btn" onClick={() => startLevelTraining(lvl.level)}>
                    {model ? 'Resume Training' : 'Start Training'}
                  </button>
                )}
                {isGpuActive && (
                  <button className="ctrl-btn stop-btn" onClick={handleStopGpuTraining}>
                    Stop Training
                  </button>
                )}
                {!isComplete && !prevComplete && <span className="level-locked">Complete Level {lvl.level - 1} first</span>}
              </div>
            );
          })}
          <p className="gpu-hint">Training runs on a remote GPU. You can close this page and come back later.</p>
        </div>
      )}

      {mode === 'admin' && (
        <div className="admin-panel">
          <h3>Saved Models</h3>
          {statusMsg && <div className="status-msg">{statusMsg}</div>}
          {savedModels.length === 0 ? (
            <p className="empty-msg">No saved models yet. Train an AI and save it first!</p>
          ) : (
            <div className="model-list">
              {savedModels.map(m => (
                <div key={m.id} className="model-card">
                  <div className="model-info">
                    <span className="model-name">{m.name}</span>
                    <span className="model-meta">{m.episodes} episodes &bull; B:{m.blue_wins} R:{m.red_wins} D:{m.draws}</span>
                    <span className="model-date">{new Date(m.created_at).toLocaleDateString()}</span>
                  </div>
                  <div className="model-actions">
                    <button className="ctrl-btn load-btn" onClick={() => handleLoad(m.id)} disabled={loadingModel}>
                      {loadingModel ? '...' : '▶ Load'}
                    </button>
                    <button className="ctrl-btn del-btn" onClick={() => handleDelete(m.id, m.name)}>
                      🗑 Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
          <button className="ctrl-btn refresh-btn" onClick={fetchModels}>↻ Refresh</button>
        </div>
      )}

      {mode === 'play-ai' && !playLevel && (
        <div className="level-select">
          <h3>Select Difficulty</h3>
          {statusMsg && <div className="status-msg">{statusMsg}</div>}
          {AI_LEVELS.map(lvl => {
            const model = levelModels[lvl.level];
            const available = !!model && model.episodes > 0;
            return (
              <div key={lvl.level} className={'level-pick-card' + (available ? '' : ' locked')}>
                <div className="level-pick-header">
                  <span className="level-name">{lvl.name}</span>
                  <span className="level-label">{lvl.label}</span>
                </div>
                <div className="level-pick-meta">
                  {available ? `${(model!.episodes / 1000).toFixed(0)}K episodes trained` : 'Not trained yet'}
                </div>
                {available ? (
                  <button className="ctrl-btn start-btn" onClick={() => selectPlayLevel(lvl.level)} disabled={loadingModel}>
                    {loadingModel ? 'Loading...' : 'Play'}
                  </button>
                ) : (
                  <span className="level-locked">Train this level first</span>
                )}
              </div>
            );
          })}
        </div>
      )}

      {isPlayMode && !isPortrait && (
        <div className="instructions">
          <h3>Controls</h3>
          <div className="instruction-grid">
            <div><kbd>Click / Tap</kbd> Set destination</div>
            <div><kbd>Space</kbd> Pause / Resume</div>
            <div><kbd>C</kbd> Attempt steal (1s cooldown)</div>
          </div>
          <div className="rules">
            <h4>Rules</h4>
            <ul>
              <li>Carry the puck into the goal zone to score</li>
              <li>After stealing, return to backcheck zone first</li>
              <li>Steal chance: 50% at 4ft, 20% at 8ft (higher if carrier comes toward you)</li>
              {mode === 'play-ai' && <li><strong>Red player is controlled by AI</strong></li>}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
