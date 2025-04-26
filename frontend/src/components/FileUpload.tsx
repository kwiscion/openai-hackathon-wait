import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/use-toast";
import { FileText, Upload } from "lucide-react";
import { useRef, useState } from "react";

interface FileUploadProps {
  onFileAccepted: (response: any) => void;
}

export const FileUpload = ({ onFileAccepted }: FileUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    validateAndProcessFile(file);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      validateAndProcessFile(file);
    }
  };

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  const validateAndProcessFile = async (file: File) => {
    if (file.type !== "application/pdf") {
      toast({
        variant: "destructive",
        title: "Invalid file type",
        description: "Please upload a PDF file",
      });
      return;
    }

    try {
      setIsUploading(true);
      // We'll just pass the file to the parent component now
      // The actual upload will happen in the Index component
      onFileAccepted(file);
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Upload failed",
        description: error instanceof Error ? error.message : "Failed to upload PDF",
      });
      setIsUploading(false);
    }
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
        isDragging ? "border-primary bg-primary/5" : "border-gray-300"
      }`}
    >
      <div className="flex flex-col items-center gap-4">
        <FileText className="w-16 h-16 text-gray-400" />
        <div className="text-xl font-medium">
          Drag and drop your PDF here
        </div>
        <p className="text-sm text-gray-500">
          or
        </p>
        <div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            onChange={handleFileInput}
            className="hidden"
            id="file-upload"
            disabled={isUploading}
          />
          <Button 
            onClick={handleButtonClick}
            disabled={isUploading}
          >
            <Upload className="mr-2" />
            {isUploading ? "Uploading..." : "Select PDF"}
          </Button>
        </div>
      </div>
    </div>
  );
};
