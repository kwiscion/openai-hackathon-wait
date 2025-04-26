import { FileUpload } from "@/components/FileUpload";
import { ProcessingState } from "@/components/ProcessingState";
import { ResultsDisplay } from "@/components/ResultsDisplay";
import { useToast } from "@/components/ui/use-toast";
import { uploadPDF } from "@/lib/api";
import { useState } from "react";

type ReviewState = "upload" | "processing" | "results";

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
  const { toast } = useToast();

  const handleFileAccepted = async (file: File) => {
    setReviewState("processing");
    
    try {
      // Upload the file to the API
      const response = await uploadPDF(file);
      
      // Set the results from the API response
      setResults(response.data);
      setReviewState("results");
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Upload failed",
        description: error instanceof Error ? error.message : "Failed to upload PDF",
      });
      setReviewState("upload");
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold text-center mb-8">
          Paper Review System
        </h1>
        
        <div className="max-w-3xl mx-auto bg-white rounded-lg shadow-md p-8">
          {reviewState === "upload" && (
            <FileUpload onFileAccepted={handleFileAccepted} />
          )}
          
          {reviewState === "processing" && <ProcessingState />}
          
          {reviewState === "results" && results && (
            <ResultsDisplay 
              feedback={results.feedback} 
              title={results.title}
              sections={results.sections}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default Index;
