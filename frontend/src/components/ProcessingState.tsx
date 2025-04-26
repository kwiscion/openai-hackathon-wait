
import { Loader } from "lucide-react";

export const ProcessingState = () => {
  return (
    <div className="flex flex-col items-center gap-4 py-12">
      <Loader className="w-12 h-12 animate-spin text-primary" />
      <h2 className="text-xl font-medium">Analyzing your paper...</h2>
      <p className="text-sm text-gray-500">This typically takes 1-2 minutes</p>
    </div>
  );
};
