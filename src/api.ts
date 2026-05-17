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
