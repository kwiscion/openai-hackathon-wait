import { API_URL, MOCK_API, mockPDFUploadResponse } from './config';

interface TaskHistory {
  step: string;
  status: string;
  message: string;
  timestamp: string;
}

interface TaskStatus {
  task_id: number;
  status: string;
  file_name: string;
  current_step?: string;
  history?: TaskHistory[];
  data?: any;
}

/**
 * Uploads a PDF file to the API and polls until processing is complete
 * @param file The PDF file to upload
 * @returns Promise with the API response
 */
export async function uploadPDF(file: File): Promise<{
  status: TaskStatus;
  data?: any;
}> {
  console.log('uploadPDF called with file:', file.name);
  console.log('Using API URL:', API_URL);
  console.log('MOCK_API setting:', MOCK_API);
  
  if (MOCK_API) {
    console.log('Using mock data');
    // Return mock data after a small delay to simulate network latency
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve(mockPDFUploadResponse);
      }, 3000); // Simulate 3 seconds of processing
    });
  }

  console.log('Preparing to send real API request');
  const formData = new FormData();
  formData.append('file', file);

  // Submit the file for processing
  console.log('Sending POST request to:', `${API_URL}/upload`);
  try {
    const uploadResponse = await fetch(`${API_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    console.log('Upload response status:', uploadResponse.status);
    
    if (!uploadResponse.ok) {
      const errorData = await uploadResponse.json().catch(() => ({}));
      console.error('Upload failed:', errorData);
      throw new Error(errorData.message || errorData.detail || 'Failed to upload PDF');
    }

    const taskStatus = await uploadResponse.json() as TaskStatus;
    console.log('Upload response data:', taskStatus);
    
    // Poll for job completion
    return pollTaskStatus(taskStatus.task_id);
  } catch (error) {
    console.error('Error in uploadPDF:', error);
    throw error;
  }
}

/**
 * Polls the task status endpoint until the task is complete
 * @param taskId The ID of the task to poll
 * @returns Promise with the API response
 */
async function pollTaskStatus(taskId: number): Promise<{
  status: TaskStatus;
  data?: any;
}> {
  console.log('Starting to poll task status for task ID:', taskId);
  const maxAttempts = 60; // Maximum number of polling attempts
  const interval = 1000; // Poll every 1 second (changed from 2 seconds)
  
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    console.log(`Poll attempt ${attempt + 1}/${maxAttempts}`);
    const response = await fetch(`${API_URL}/status/${taskId}`);
    
    console.log('Status response:', response.status);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.error('Status check failed:', errorData);
      throw new Error(errorData.message || errorData.detail || 'Failed to check task status');
    }
    
    const taskStatus = await response.json() as TaskStatus;
    console.log('Status data:', taskStatus);
    
    // Update progress callbacks if provided
    if (taskStatus.status === "completed" && taskStatus.data) {
      console.log('Task completed successfully');
      return {
        status: taskStatus,
        data: taskStatus.data
      };
    }
    
    console.log(`Task still processing. Waiting ${interval}ms before next check.`);
    // Wait before polling again
    await new Promise(resolve => setTimeout(resolve, interval));
  }
  
  console.error('Task timed out after maximum polling attempts');
  throw new Error('Processing timed out');
} 