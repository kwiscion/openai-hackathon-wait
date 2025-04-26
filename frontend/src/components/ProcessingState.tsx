import { CheckCircle, Clock, Loader } from "lucide-react";
import { useEffect, useState } from "react";

interface TaskHistory {
  step: string;
  status: string;
  message: string;
  timestamp: string;
}

interface ProcessingStateProps {
  history?: TaskHistory[];
}

// Mapping of step names to readable titles
const stepTitles: Record<string, string> = {
  "upload": "Upload File",
  "gathering_papers": "Gathering Relevant Papers",
  "setting_up_reviews": "Setting Up Review Framework",
  "generating_reviews": "Generating Section Reviews",
  "meta_review": "Compiling Final Assessment"
};

export const ProcessingState = ({ history = [] }: ProcessingStateProps) => {
  // Force re-render counter
  const [, setForceUpdate] = useState(0);
  
  // Add effect to re-render component every 2 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setForceUpdate(prev => prev + 1);
    }, 2000);
    
    return () => clearInterval(interval);
  }, [history.length]); // Re-setup the interval if history length changes
  
  // Explicitly make a local copy of history on each render
  const historySnapshot = [...history];
  
  // Sort history by timestamp (oldest first)
  const sortedHistory = historySnapshot.sort(
    (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
  );
  
  // Get the latest status for each step
  const latestStatusByStep: Record<string, string> = {};
  for (const entry of sortedHistory) {
    latestStatusByStep[entry.step] = entry.status;
  }
  
  // Get the current active step (most recent in_progress or the last step)
  const activeStep = sortedHistory.length > 0 
    ? sortedHistory[sortedHistory.length - 1].step 
    : '';
  
  return (
    <div className="flex flex-col items-center gap-6 py-8 w-full max-w-lg mx-auto">
      <Loader className="w-12 h-12 animate-spin text-primary" />
      <h2 className="text-xl font-medium">Analyzing your paper...</h2>
      
      {/* Show progress log */}
      <div className="w-full mt-4 space-y-4">
        {sortedHistory.map((entry, index) => {
          const isActive = entry.step === activeStep;
          const isCompleted = latestStatusByStep[entry.step] === 'completed';
          const isInProgress = latestStatusByStep[entry.step] === 'in_progress';
          
          return (
            <div 
              key={`${entry.step}-${index}-${entry.timestamp}-${Date.now()}`}
              className={`
                p-3 rounded-md border 
                ${isActive ? 'bg-primary/10 border-primary' : 'border-gray-200'}
              `}
            >
              <div className="flex items-center gap-2 mb-1">
                {isCompleted ? (
                  <CheckCircle className="h-4 w-4 text-green-500" />
                ) : isInProgress ? (
                  <Loader className="h-4 w-4 text-primary animate-spin" />
                ) : (
                  <Clock className="h-4 w-4 text-gray-400" />
                )}
                <h3 className={`text-sm font-medium ${isActive ? 'text-primary' : ''}`}>
                  {stepTitles[entry.step] || entry.step}
                </h3>
                <span className="text-xs text-gray-500 ml-auto">
                  {new Date(entry.timestamp).toLocaleTimeString()}
                </span>
              </div>
              
              <p className="text-sm text-gray-600 ml-6">
                {entry.message}
              </p>
            </div>
          );
        })}
      </div>
      
      {sortedHistory.length === 0 && (
        <div className="text-center text-gray-500 italic">
          Initializing analysis...
        </div>
      )}
      
      <p className="text-sm text-gray-500 mt-2">
        This typically takes 1-2 minutes. Please don't close this page.
      </p>
      
      <button 
        onClick={() => {
          localStorage.clear();
          window.location.reload();
        }}
        className="mt-4 text-sm text-red-500 hover:text-red-700 underline"
      >
        Reset Application
      </button>
    </div>
  );
};