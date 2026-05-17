const API_URL = import.meta.env.VITE_API_URL || 'https://hockey-api-zrey.onrender.com';

export interface SavedModel {
  id: string;
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

export async function getModel(id: string): Promise<SavedModel> {
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

export async function updateModel(id: string, data: {
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

export async function deleteModel(id: string): Promise<void> {
  const res = await fetch(`${API_URL}/models/${id}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete model');
}

// ── GPU Training ──

export interface TrainingStatus {
  status: string;
  pod_id?: string;
  model_name?: string;
  episode?: number;
  total_episodes?: number;
  blue_wins?: number;
  red_wins?: number;
  draws?: number;
  eps_per_sec?: number;
  cost_per_hr?: number;
  started_at?: string;
  last_report?: string;
}

export async function startGPUTraining(data: {
  model_name: string;
  episodes: number;
  save_interval?: number;
  gpu_type?: string;
}): Promise<{ pod_id: string; cost_per_hr: number; status: string }> {
  const res = await fetch(`${API_URL}/training/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Failed to start GPU training: ${err}`);
  }
  return res.json();
}

export async function stopGPUTraining(): Promise<void> {
  const res = await fetch(`${API_URL}/training/stop`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to stop GPU training');
}

export async function getTrainingStatus(): Promise<TrainingStatus> {
  const res = await fetch(`${API_URL}/training/status`);
  if (!res.ok) throw new Error('Failed to get training status');
  return res.json();
}
