import { useRef, useEffect, useState, useCallback } from 'react';
import { GameState } from './game/types';
import {
  RINK_WIDTH, RINK_HEIGHT, FIXED_DT, PLAYER_RADIUS,
} from './game/constants';
import { createInitialState, gameStep, attemptSteal } from './game/physics';
import { render } from './game/renderer';
import { AIAgent, PolicyNetwork, NETWORK_SIZES } from './game/ai';
import { Trainer, TrainingStats } from './game/training';
import { listModels, getModel, saveModel, deleteModel, SavedModel, startGPUTraining, stopGPUTraining, getTrainingStatus, TrainingStatus } from './api';
import './App.css';

type GameMode = 'practice' | 'training' | 'play-ai' | 'admin';

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<GameState>(createInitialState(0));
  const rafRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);
  const accumulatorRef = useRef<number>(0);
  const scaleRef = useRef<number>(1);
  const offsetRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  const trainerRef = useRef<Trainer | null>(null);
  const aiAgentRef = useRef<AIAgent | null>(null);
  const networkRef = useRef<PolicyNetwork | null>(null);

  const [mode, setMode] = useState<GameMode>('practice');
  const [score, setScore] = useState<[number, number]>([0, 0]);
  const [paused, setPaused] = useState(false);
  const [trainingRunning, setTrainingRunning] = useState(false);
  const [trainingStats, setTrainingStats] = useState<TrainingStats | null>(null);
  const [trainingSpeed, setTrainingSpeed] = useState(5);
  const [trainingRinks, setTrainingRinks] = useState(1);
  const [savedModels, setSavedModels] = useState<SavedModel[]>([]);
  const [saveName, setSaveName] = useState('');
  const [saving, setSaving] = useState(false);
  const [loadingModel, setLoadingModel] = useState(false);
  const [statusMsg, setStatusMsg] = useState('');
  const [gpuStatus, setGpuStatus] = useState<TrainingStatus | null>(null);
  const [gpuModelName, setGpuModelName] = useState('gpu-trained');
  const [gpuEpisodes, setGpuEpisodes] = useState(100000);
  const [gpuStarting, setGpuStarting] = useState(false);
  const gpuPollRef = useRef<number>(0);

  const PADDING = 10;

  const resizeCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const vh = window.innerHeight;
    const vw = window.innerWidth;
    const isMobile = vw <= 900;
    const reservedHeight = isMobile ? 80 : 160;
    const availableHeight = vh - reservedHeight;
    const availableWidth = Math.min(container.clientWidth, 1100) - PADDING * 2;
    const scaleByWidth = availableWidth / RINK_WIDTH;
    const scaleByHeight = (availableHeight - PADDING * 2) / RINK_HEIGHT;
    const s = Math.min(scaleByWidth, scaleByHeight);
    const cw = RINK_WIDTH * s + PADDING * 2;
    const ch = RINK_HEIGHT * s + PADDING * 2;
    const dpr = window.devicePixelRatio || 1;
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
    const x = (clientX - rect.left - offsetRef.current.x) / scaleRef.current;
    const y = (clientY - rect.top - offsetRef.current.y) / scaleRef.current;
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
    } catch {
      console.error('Failed to fetch models');
    }
  }, []);

  const handleSave = useCallback(async () => {
    const net = networkRef.current;
    if (!net || !saveName.trim()) return;
    setSaving(true);
    setStatusMsg('');
    try {
      const stats = trainingStats || { episode: 0, blueWins: 0, redWins: 0, draws: 0 };
      await saveModel({
        name: saveName.trim(),
        weights: net.serialize(),
        episodes: stats.episode,
        blue_wins: stats.blueWins,
        red_wins: stats.redWins,
        draws: stats.draws,
      });
      setSaveName('');
      setStatusMsg('Model saved!');
      await fetchModels();
    } catch {
      setStatusMsg('Failed to save model');
    }
    setSaving(false);
  }, [saveName, trainingStats, fetchModels]);

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

  // GPU Training
  const pollGpuStatus = useCallback(async () => {
    try {
      const status = await getTrainingStatus();
      setGpuStatus(status);
      if (status.status === 'completed' || status.status === 'stopped' || status.status === 'idle') {
        clearInterval(gpuPollRef.current);
        gpuPollRef.current = 0;
      }
    } catch { /* ignore */ }
  }, []);

  const handleStartGPU = useCallback(async () => {
    if (!gpuModelName.trim()) return;
    setGpuStarting(true);
    setStatusMsg('');
    try {
      const result = await startGPUTraining({
        model_name: gpuModelName.trim(),
        episodes: gpuEpisodes,
      });
      setStatusMsg(`GPU pod starting... ($${result.cost_per_hr}/hr)`);
      setGpuStatus({ status: 'starting', model_name: gpuModelName, total_episodes: gpuEpisodes });
      if (gpuPollRef.current) clearInterval(gpuPollRef.current);
      gpuPollRef.current = window.setInterval(pollGpuStatus, 5000);
    } catch (e) {
      setStatusMsg(`Failed to start GPU: ${e instanceof Error ? e.message : 'unknown error'}`);
    }
    setGpuStarting(false);
  }, [gpuModelName, gpuEpisodes, pollGpuStatus]);

  const handleStopGPU = useCallback(async () => {
    try {
      await stopGPUTraining();
      setStatusMsg('GPU training stopped');
      if (gpuPollRef.current) clearInterval(gpuPollRef.current);
      setGpuStatus(prev => prev ? { ...prev, status: 'stopped' } : null);
    } catch {
      setStatusMsg('Failed to stop GPU training');
    }
  }, []);

  // Mode switching
  const switchMode = useCallback((newMode: GameMode) => {
    if (trainerRef.current) trainerRef.current.stop();
    cancelAnimationFrame(rafRef.current);
    lastTimeRef.current = 0;
    accumulatorRef.current = 0;
    if (newMode === 'practice' || newMode === 'play-ai') {
      stateRef.current = createInitialState(0);
      setScore([0, 0]);
      setPaused(false);
      if (newMode === 'play-ai') {
        const net = networkRef.current || new PolicyNetwork(NETWORK_SIZES);
        networkRef.current = net;
        aiAgentRef.current = new AIAgent(net);
      }
    }
    if (newMode === 'training') {
      if (!trainerRef.current) {
        trainerRef.current = new Trainer(undefined, networkRef.current || undefined);
        networkRef.current = trainerRef.current.getNetwork();
      }
    }
    if (newMode === 'admin') {
      fetchModels();
    }
    setMode(newMode);
    setTrainingRunning(false);
    setStatusMsg('');
  }, [fetchModels]);

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
          render(ctx, state, canvas.width / dpr, canvas.height / dpr, scaleRef.current, offsetRef.current.x, offsetRef.current.y);
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
            render(ctx, trainer.currentState, canvas.width / dpr, canvas.height / dpr, scaleRef.current, offsetRef.current.x, offsetRef.current.y);
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

  // Training controls
  const toggleTraining = useCallback(() => {
    const trainer = trainerRef.current;
    if (!trainer) return;
    if (trainer.running) {
      trainer.stop();
      setTrainingRunning(false);
    } else {
      trainer.speed = trainingSpeed;
      trainer.rinks = trainingRinks;
      trainer.onStatsUpdate = (s) => setTrainingStats({ ...s });
      trainer.start();
      setTrainingRunning(true);
    }
  }, [trainingSpeed, trainingRinks]);

  const handleSpeedChange = useCallback((val: number) => {
    setTrainingSpeed(val);
    if (trainerRef.current) trainerRef.current.speed = val;
  }, []);

  const handleRinksChange = useCallback((val: number) => {
    setTrainingRinks(val);
    if (trainerRef.current) trainerRef.current.rinks = val;
  }, []);

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
    const touch = e.touches[0];
    if (touch) setBlueDestination(touch.clientX, touch.clientY);
  };

  const isPlayMode = mode === 'practice' || mode === 'play-ai';
  const isGameMode = mode === 'practice' || mode === 'play-ai' || mode === 'training';
  const stats = trainingStats;

  return (
    <div className="app">
      <div className="landscape-prompt">
        <div className="landscape-prompt-content">
          <span className="rotate-icon">&#x1F504;</span>
          <p>Rotate your phone to landscape</p>
        </div>
      </div>

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

      {mode === 'training' && stats && (
        <div className="training-stats">
          <div className="stat"><span className="stat-label">Episodes</span><span className="stat-value">{stats.episode}</span></div>
          <div className="stat"><span className="stat-label">Blue W</span><span className="stat-value blue">{stats.blueWins}</span></div>
          <div className="stat"><span className="stat-label">Red W</span><span className="stat-value red">{stats.redWins}</span></div>
          <div className="stat"><span className="stat-label">Draws</span><span className="stat-value">{stats.draws}</span></div>
          <div className="stat"><span className="stat-label">Goals/ep</span><span className="stat-value">{stats.avgGoalsPerEp.toFixed(1)}</span></div>
        </div>
      )}

      {isGameMode && (
        <div className="canvas-container" ref={containerRef}>
          <canvas ref={canvasRef} onClick={handleCanvasClick} onTouchStart={handleCanvasTouch} />
        </div>
      )}

      {isPlayMode && (
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
        <div className="training-controls">
          <button className={'ctrl-btn ' + (trainingRunning ? 'stop-btn' : 'start-btn')} onClick={toggleTraining}>
            {trainingRunning ? '\u23F9 STOP' : '\u25B6 TRAIN'}
          </button>
          <div className="speed-control">
            <label>Rinks: {trainingRinks}</label>
            <input type="range" min={1} max={50} value={trainingRinks} onChange={(e) => handleRinksChange(Number(e.target.value))} />
          </div>
          <div className="speed-control">
            <label>Speed: {trainingSpeed}x</label>
            <input type="range" min={1} max={50} value={trainingSpeed} onChange={(e) => handleSpeedChange(Number(e.target.value))} />
          </div>
        </div>
      )}

      {mode === 'training' && (
        <div className="save-bar">
          <input
            type="text"
            className="save-input"
            placeholder="Model name..."
            value={saveName}
            onChange={(e) => setSaveName(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') handleSave(); }}
          />
          <button className="ctrl-btn save-btn" onClick={handleSave} disabled={saving || !saveName.trim() || !networkRef.current}>
            {saving ? 'Saving...' : 'Save'}
          </button>
          {statusMsg && <span className="status-msg">{statusMsg}</span>}
        </div>
      )}

      {mode === 'training' && (
        <div className="gpu-panel">
          <h3>GPU Training (RunPod A100)</h3>
          {(!gpuStatus || gpuStatus.status === 'idle' || gpuStatus.status === 'stopped' || gpuStatus.status === 'completed') ? (
            <div className="gpu-start-form">
              <div className="gpu-form-row">
                <label>Model Name</label>
                <input
                  type="text"
                  className="save-input"
                  value={gpuModelName}
                  onChange={(e) => setGpuModelName(e.target.value)}
                />
              </div>
              <div className="gpu-form-row">
                <label>Episodes</label>
                <select className="gpu-select" value={gpuEpisodes} onChange={(e) => setGpuEpisodes(Number(e.target.value))}>
                  <option value={10000}>10,000</option>
                  <option value={50000}>50,000</option>
                  <option value={100000}>100,000</option>
                  <option value={500000}>500,000</option>
                  <option value={1000000}>1,000,000</option>
                </select>
              </div>
              <button
                className="ctrl-btn gpu-start-btn"
                onClick={handleStartGPU}
                disabled={gpuStarting || !gpuModelName.trim()}
              >
                {gpuStarting ? 'Starting...' : 'Train on GPU (~$3/hr)'}
              </button>
              {gpuStatus?.status === 'completed' && <span className="status-msg gpu-done">Training completed!</span>}
              {gpuStatus?.status === 'stopped' && <span className="status-msg">Training stopped</span>}
            </div>
          ) : (
            <div className="gpu-monitor">
              <div className="gpu-status-badge">{gpuStatus.status === 'starting' ? 'Starting pod...' : 'Training'}</div>
              {gpuStatus.episode !== undefined && (
                <div className="gpu-stats">
                  <div className="stat">
                    <span className="stat-label">Progress</span>
                    <span className="stat-value">{gpuStatus.episode?.toLocaleString()} / {gpuStatus.total_episodes?.toLocaleString()}</span>
                  </div>
                  <div className="stat">
                    <span className="stat-label">Blue W</span>
                    <span className="stat-value blue">{gpuStatus.blue_wins?.toLocaleString()}</span>
                  </div>
                  <div className="stat">
                    <span className="stat-label">Red W</span>
                    <span className="stat-value red">{gpuStatus.red_wins?.toLocaleString()}</span>
                  </div>
                  <div className="stat">
                    <span className="stat-label">Draws</span>
                    <span className="stat-value">{gpuStatus.draws?.toLocaleString()}</span>
                  </div>
                  <div className="stat">
                    <span className="stat-label">Speed</span>
                    <span className="stat-value">{gpuStatus.eps_per_sec?.toFixed(1)} ep/s</span>
                  </div>
                  {gpuStatus.cost_per_hr && (
                    <div className="stat">
                      <span className="stat-label">Cost</span>
                      <span className="stat-value">${gpuStatus.cost_per_hr}/hr</span>
                    </div>
                  )}
                </div>
              )}
              <div className="gpu-progress-bar">
                <div className="gpu-progress-fill" style={{ width: `${Math.min(100, ((gpuStatus.episode || 0) / (gpuStatus.total_episodes || 1)) * 100)}%` }} />
              </div>
              <div className="gpu-actions">
                <button className="ctrl-btn load-btn" onClick={() => { fetchModels(); switchMode('admin'); }}>
                  Load Latest
                </button>
                <button className="ctrl-btn stop-btn" onClick={handleStopGPU}>
                  Stop Training
                </button>
              </div>
            </div>
          )}
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

      {isPlayMode && (
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
