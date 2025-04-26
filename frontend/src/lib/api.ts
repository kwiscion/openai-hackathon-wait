import { API_URL, MOCK_API, mockPDFUploadResponse } from './config';

/**
 * Uploads a PDF file to the API
 * @param file The PDF file to upload
 * @returns Promise with the API response
 */
export async function uploadPDF(file: File) {
  if (MOCK_API) {
    // Return mock data after a small delay to simulate network latency
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve(mockPDFUploadResponse);
      }, 3000); // Simulate 3 seconds of processing
    });
  }

  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_URL}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.message || 'Failed to upload PDF');
  }

  return await response.json();
} 