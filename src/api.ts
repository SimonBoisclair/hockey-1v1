const API_URL = import.meta.env.VITE_API_URL || 'https://hockey-api-zrey.onrender.com';

export interface SavedModel {
  id: number;
  name: string;
  episodes: number;
  blue_wins: number;
  red_wins: number;
  draws: number;
  created_at: string;
  updated_at: string;
  weights?: string;
}

export async function listModels(): Promise<SavedModel[]> {
  const res = await fetch(`${API_URL}/models`);
  if (!res.ok) throw new Error('Failed to list models');
  return res.json();
}

export async function getModel(id: number): Promise<SavedModel> {
  const res = await fetch(`${API_URL}/models/${id}`);
  if (!res.ok) throw new Error('Failed to get model');
  return res.json();
}

export async function saveModel(data: {
  name: string;
  weights: string;
  episodes: number;
  blue_wins: number;
  red_wins: number;
  draws: number;
}): Promise<SavedModel> {
  const res = await fetch(`${API_URL}/models`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error('Failed to save model');
  return res.json();
}

export async function updateModel(id: number, data: {
  name?: string;
  weights?: string;
  episodes?: number;
  blue_wins?: number;
  red_wins?: number;
  draws?: number;
}): Promise<SavedModel> {
  const res = await fetch(`${API_URL}/models/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error('Failed to update model');
  return res.json();
}

export async function deleteModel(id: number): Promise<void> {
  const res = await fetch(`${API_URL}/models/${id}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete model');
}

// ── Training API ──

export interface TrainingStatus {
  status: string;
  pod_id?: string;
  model_name?: string;
  total_episodes?: number;
  episode?: number;
  blue_wins?: number;
  red_wins?: number;
  draws?: number;
  eps_per_sec?: number;
  cost_per_hr?: number;
  started_at?: string;
  level?: number | null;
}

export async function startGpuTraining(data: {
  model_name: string;
  episodes: number;
  save_interval?: number;
  level?: number;
  load_model_id?: string;
  compat_mode?: boolean;
}): Promise<{ pod_id: string; cost_per_hr: number; status: string; gpu_type: string }> {
  const res = await fetch(`${API_URL}/training/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to start training' }));
    throw new Error(err.detail || 'Failed to start training');
  }
  return res.json();
}

export async function stopGpuTraining(): Promise<{ status: string }> {
  const res = await fetch(`${API_URL}/training/stop`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to stop training');
  return res.json();
}

export async function getTrainingStatus(): Promise<TrainingStatus> {
  const res = await fetch(`${API_URL}/training/status`);
  if (!res.ok) throw new Error('Failed to get training status');
  return res.json();
}
