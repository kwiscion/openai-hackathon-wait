import { FileUpload } from "@/components/FileUpload";
import { ProcessingState } from "@/components/ProcessingState";
import { ResultsDisplay } from "@/components/ResultsDisplay";
import { useToast } from "@/components/ui/use-toast";
import { uploadPDF } from "@/lib/api";
import { useEffect, useRef, useState } from "react";

type ReviewState = "upload" | "processing" | "results";

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
  data?: {
    data?: any;
    [key: string]: any;
  };
}

interface ReviewResult {
  title: string;
  feedback: string;
  sections?: {
    title: string;
    score: number;
    feedback: string;
  }[];
}

const Index = () => {
  const [reviewState, setReviewState] = useState<ReviewState>("upload");
  const [results, setResults] = useState<ReviewResult | null>(null);
  const [taskStatus, setTaskStatus] = useState<TaskStatus | null>(null);
  const { toast } = useToast();
  
  // Polling interval reference
  const pollingIntervalRef = useRef<number | null>(null);
  
  // Add event listeners for page unload and visibility change
  useEffect(() => {
    // Helper to save current state
    const saveCurrentState = () => {
      console.log('Saving current state before page close/refresh:', { reviewState, taskStatus });
      if (reviewState === 'processing' && taskStatus?.task_id) {
        localStorage.setItem('reviewState', reviewState);
        localStorage.setItem('currentTaskId', taskStatus.task_id.toString());
      }
    };
    
    // Save state before page is unloaded
    const handleBeforeUnload = () => {
      saveCurrentState();
    };
    
    // Save state when page visibility changes (especially when browser tab is about to close)
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'hidden') {
        saveCurrentState();
      }
    };
    
    // Add event listeners
    window.addEventListener('beforeunload', handleBeforeUnload);
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    // Clean up
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [reviewState, taskStatus]);
  
  // Restore state from localStorage on mount
  useEffect(() => {
    try {
      // Check what's stored in localStorage
      const currentState = localStorage.getItem('reviewState') as ReviewState | null;
      const savedTaskId = localStorage.getItem('currentTaskId');
      const savedResults = localStorage.getItem('results');
      const taskCompleted = localStorage.getItem('taskCompleted') === 'true';
      
      console.log('State from localStorage:', { 
        currentState, 
        savedTaskId, 
        hasResults: !!savedResults,
        taskCompleted
      });
      
      // IMPORTANT: Start by setting the UI state according to localStorage
      // This ensures the user sees something right away instead of defaulting to upload
      if (currentState) {
        console.log('Setting initial UI state from localStorage:', currentState);
        setReviewState(currentState);
        
        // If we have results stored, use them
        if (currentState === 'results' && savedResults) {
          try {
            const parsedResults = JSON.parse(savedResults);
            console.log('Restored results from localStorage');
            setResults(parsedResults);
          } catch (e) {
            console.error('Error parsing saved results:', e);
          }
        }
      }
      
      // If there's a task ID, check its status regardless of current UI state
      if (savedTaskId) {
        const taskId = parseInt(savedTaskId, 10);
        if (!isNaN(taskId)) {
          console.log('Found task ID in localStorage, checking status:', taskId);
          
          // Fetch current task status
          fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/status/${taskId}`)
            .then(response => {
              console.log('Status API response:', response.status);
              if (!response.ok) throw new Error(`Status API returned ${response.status}`);
              return response.json();
            })
            .then(taskData => {
              console.log('Task status API returned:', taskData);
              
              // Update task status in UI
              setTaskStatus({ 
                ...taskData, 
                history: taskData.history ? [...taskData.history] : [] 
              });
              
              // Handle completed task
              if (taskData.status === 'completed' && taskData.data) {
                console.log('Task is completed with data');
                setResults(taskData.data.data);
                localStorage.setItem('results', JSON.stringify(taskData.data.data));
                localStorage.setItem('taskCompleted', 'true');
                
                // If not already in results view, show processing briefly then go to results
                if (currentState !== 'results') {
                  setReviewState('processing'); // Show processing view briefly
                  setTimeout(() => {
                    setReviewState('results');
                    localStorage.setItem('reviewState', 'results');
                  }, 2000);
                }
              }
              // Handle processing task
              else if (taskData.status === 'processing') {
                console.log('Task is still processing, resuming polling');
                setReviewState('processing');
                localStorage.setItem('reviewState', 'processing');
                startPolling(taskId);
              }
              // Handle other states
              else {
                console.log('Task in state:', taskData.status);
                if (currentState === 'processing') {
                  // Start polling anyway to see if we get updates
                  startPolling(taskId);
                }
              }
            })
            .catch(error => {
              console.error('Error checking task status:', error);
              // IMPORTANT: If API fails, keep current UI state
              // If we're in processing state, try to restart polling
              if (currentState === 'processing') {
                console.log('API error, but attempting to restart polling');
                startPolling(taskId);
              }
            });
        }
      }
      // No saved state at all - only now default to upload
      else if (!currentState) {
        console.log('No state in localStorage - defaulting to upload');
        setReviewState('upload');
      }
    } catch (error) {
      console.error('Serious error in state restoration logic:', error);
      // We only reset on catastrophic error in our logic
      setReviewState('upload');
    }
  }, []);
  
  // Helper function to reset to upload state
  const resetToUpload = () => {
    console.log('Resetting to upload state');
    setReviewState('upload');
    localStorage.clear();
  };
  
  // Clean up polling on unmount
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        window.clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  // Log changes to history
  useEffect(() => {
    if (taskStatus?.history) {
      console.log(`History array changed: ${taskStatus.history.length} items. Current step: ${taskStatus.current_step}`);
    }
  }, [taskStatus?.history]);

  // Special effect to handle poll recovery if the page was refreshed during processing
  useEffect(() => {
    // Only run on initial mount
    const savedState = localStorage.getItem('reviewState');
    const savedTaskId = localStorage.getItem('currentTaskId');
    
    if (savedState === 'processing' && savedTaskId && reviewState === 'processing') {
      const taskId = parseInt(savedTaskId, 10);
      if (!isNaN(taskId) && !pollingIntervalRef.current) {
        console.log('Recovery: Ensuring polling is active for task:', taskId);
        // Wait a moment to make sure any existing polling has started
        const recoverTimeout = setTimeout(() => {
          if (!pollingIntervalRef.current) {
            console.log('Recovery: Restarting polling that was lost');
            startPolling(taskId);
          }
        }, 2000);
        
        return () => clearTimeout(recoverTimeout);
      }
    }
  }, [reviewState]);

  const handleFileAccepted = async (file: File) => {
    console.log("File accepted by Index component:", file.name);
    
    // First clear any existing state
    localStorage.clear();
    
    // Update state and save to localStorage
    setReviewState("processing");
    localStorage.setItem('reviewState', 'processing');
    
    try {
      // Upload the file to the API
      console.log("Calling uploadPDF function");
      const response = await uploadPDF(file);
      
      console.log("API response received:", response);
      
      // Update task status
      if (response.status) {
        console.log("Setting task status:", response.status);
        setTaskStatus(response.status);
        
        // Save task ID to localStorage for persistence
        localStorage.setItem('currentTaskId', response.status.task_id.toString());
        
        // If still processing, start polling for updates immediately
        if (response.status.status === "processing") {
          console.log("Starting polling for task:", response.status.task_id);
          startPolling(response.status.task_id);
        }
      }
      
      // Set the results from the API response if completed
      if (response.data && response.status.status === "completed") {
        setResults(response.data.data);
        
        // Add a delay to show final state before transitioning to results
        console.log("Initial upload already completed, showing final state...");
        setTimeout(() => {
          setReviewState("results");
          
          // Save completed state
          localStorage.setItem('reviewState', 'results');
          localStorage.setItem('results', JSON.stringify(response.data.data));
          
          stopPolling();
        }, 2500);
      }
    } catch (error) {
      console.error("Error in handleFileAccepted:", error);
      toast({
        variant: "destructive",
        title: "Upload failed",
        description: error instanceof Error ? error.message : "Failed to upload PDF",
      });
      setReviewState("upload");
      localStorage.removeItem('currentTaskId');
      localStorage.removeItem('reviewState');
      localStorage.removeItem('results');
      stopPolling();
    }
  };
  
  // Start polling for status updates
  const startPolling = (taskId: number) => {
    console.log(`---POLLING START--- Task ID: ${taskId}`, new Date().toLocaleTimeString());
    stopPolling(); // Clear any existing polling
    
    // Ensure task ID is saved to localStorage (defensive programming)
    localStorage.setItem('currentTaskId', taskId.toString());
    localStorage.setItem('reviewState', 'processing');
    localStorage.setItem('pollingActive', 'true');
    
    // First, immediately fetch the current status
    fetchCurrentStatus();
    
    // Function to fetch current status that can be called recursively
    async function fetchCurrentStatus() {
      try {
        console.log(`Fetching status for task ${taskId}`, new Date().toLocaleTimeString());
        
        // Make the API call
        const apiUrl = `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/status/${taskId}`;
        console.log(`API Call: ${apiUrl}`);
        
        const response = await fetch(apiUrl);
        
        console.log(`API Response code: ${response.status}`);
        
        if (!response.ok) {
          console.error('Failed to fetch status update:', response.status);
          // Continue polling even if there's an error
          scheduleNextPoll();
          return;
        }
        
        const updatedStatus = await response.json() as TaskStatus;
        
        // Create a new object to ensure React detects the change
        const newStatus = { 
          ...updatedStatus, 
          history: updatedStatus.history ? [...updatedStatus.history] : [] 
        };
        
        console.log("Status update received:", {
          task_id: updatedStatus.task_id,
          status: updatedStatus.status,
          current_step: updatedStatus.current_step,
          history_length: updatedStatus.history?.length || 0,
          timestamp: new Date().toLocaleTimeString()
        });
        
        // Important: Always update the state with a new object to ensure React detects the change
        setTaskStatus(newStatus);
        
        // If task is completed, show results and stop polling
        if (updatedStatus.status === "completed" && updatedStatus.data) {
          console.log("Task completed, processing results...");
          setResults(updatedStatus.data.data);
          
          // Set a flag in localStorage that the task is complete
          localStorage.setItem('taskCompleted', 'true');
          localStorage.setItem('results', JSON.stringify(updatedStatus.data.data));
          
          // Add a delay to show final state before transitioning to results
          console.log("Task completed, showing final steps before results...");
          setTimeout(() => {
            setReviewState("results");
            localStorage.setItem('reviewState', 'results');
            localStorage.removeItem('pollingActive');
            stopPolling();
          }, 2500); // 2.5 second delay
          
          return; // Don't schedule next poll
        }
        
        // Schedule next poll if not completed
        scheduleNextPoll();
      } catch (error) {
        console.error("Error fetching status:", error);
        // Continue polling even if there's an error
        scheduleNextPoll();
      }
    }
    
    // Schedule the next poll
    function scheduleNextPoll() {
      // Always schedule the next poll if we're in the processing state
      if (pollingIntervalRef.current) {
        window.clearTimeout(pollingIntervalRef.current);
      }
      
      // We'll continue polling regardless of the localStorage flag
      // as long as we're in the processing state
      console.log('Scheduling next poll in 1 second');
      pollingIntervalRef.current = window.setTimeout(() => {
        // Double check if we should still be polling
        const reviewState = localStorage.getItem('reviewState');
        console.log(`Next poll check - Current review state: ${reviewState}`);
        
        if (reviewState === 'processing') {
          console.log('Still in processing state, continuing polling');
          fetchCurrentStatus();
        } else {
          console.log('No longer in processing state, stopping polling');
          stopPolling();
        }
      }, 1000); // Poll every second
    }
  };
  
  // Stop polling
  const stopPolling = () => {
    console.log("---POLLING STOP---", new Date().toLocaleTimeString());
    if (pollingIntervalRef.current) {
      window.clearTimeout(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  };

  const handleGoBack = () => {
    setReviewState("upload");
    setTaskStatus(null);
    stopPolling();
    
    // Clear stored state when going back to upload
    localStorage.removeItem('currentTaskId');
    localStorage.removeItem('reviewState');
    localStorage.removeItem('results');
  };

  // Add a debug component
  const DebugInfo = ({ taskStatus }: { taskStatus: TaskStatus | null }) => {
    const [showDebug, setShowDebug] = useState(false);
    
    // Only show in development
    if (import.meta.env.MODE !== 'development') return null;
    
    return (
      <div className="mt-4 border-t pt-4">
        <button 
          onClick={() => setShowDebug(!showDebug)}
          className="text-xs text-gray-500 underline"
        >
          {showDebug ? 'Hide' : 'Show'} Debug Info
        </button>
        
        {showDebug && (
          <div className="mt-2 p-2 bg-gray-100 rounded text-xs font-mono overflow-auto">
            <div>
              <strong>localStorage:</strong>
              <pre>{JSON.stringify({
                reviewState: localStorage.getItem('reviewState'),
                currentTaskId: localStorage.getItem('currentTaskId'),
                hasResults: !!localStorage.getItem('results')
              }, null, 2)}</pre>
            </div>
            <div className="mt-2">
              <strong>Task Status:</strong>
              <pre>{JSON.stringify(taskStatus, null, 2)}</pre>
            </div>
          </div>
        )}
      </div>
    );
  };

  // Log ProcessingState rendering for debugging
  useEffect(() => {
    if (reviewState === "processing" && taskStatus) {
      console.log(`ProcessingState key: process-history-${taskStatus.history?.length || 0}-${Date.now()}`);
    }
  }, [reviewState, taskStatus]);

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold text-center mb-8">
          Paper Review System
        </h1>
        
        {import.meta.env.MODE === 'development' && (
          <div className="max-w-3xl mx-auto mb-4 p-2 bg-blue-50 border border-blue-200 rounded text-xs">
            <p className="font-medium text-blue-800">
              Current State: {reviewState} | 
              Task ID: {taskStatus?.task_id} | 
              Current Step: {taskStatus?.current_step} |
              Status: {taskStatus?.status} |
              Time: {new Date().toLocaleTimeString()}
            </p>
            <div className="flex gap-4 mt-1">
              <button 
                onClick={() => {
                  localStorage.clear();
                  window.location.reload();
                }}
                className="text-red-500 underline"
              >
                Reset App
              </button>
              <button 
                onClick={async () => {
                  try {
                    const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/reset`);
                    if (response.ok) {
                      localStorage.clear();
                      alert('Database reset successful');
                      window.location.reload();
                    } else {
                      alert('Failed to reset database');
                    }
                  } catch (error) {
                    alert(`Error: ${error}`);
                  }
                }}
                className="text-red-500 underline"
              >
                Reset DB
              </button>
            </div>
          </div>
        )}
        
        <div className="max-w-3xl mx-auto bg-white rounded-lg shadow-md p-8">
          {reviewState === "upload" && (
            <FileUpload onFileAccepted={handleFileAccepted} />
          )}
          
          {reviewState === "processing" && (
            <ProcessingState 
              key={`process-history-${taskStatus?.history?.length || 0}-${Date.now()}`} 
              history={taskStatus?.history || []} 
            />
          )}
          
          {reviewState === "results" && results && (
            <ResultsDisplay 
              feedback={results.feedback} 
              title={results.title}
              sections={results.sections}
              onGoBack={handleGoBack}
            />
          )}
          
          <DebugInfo taskStatus={taskStatus} />
        </div>
      </div>
    </div>
  );
};

export default Index;
