import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ArrowLeft, Download, FileText } from "lucide-react";

interface Section {
  title: string;
  score: number;
  feedback: string;
}

interface ResultsDisplayProps {
  feedback: string;
  sections?: Section[];
  title?: string;
  onGoBack?: () => void;
}

export const ResultsDisplay = ({ 
  feedback, 
  sections = [], 
  title = "Review Results",
  onGoBack
}: ResultsDisplayProps) => {
  const handleExport = () => {
    const element = document.createElement("a");
    const file = new Blob([feedback], { type: "text/plain" });
    element.href = URL.createObjectURL(file);
    element.download = "paper-review.txt";
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  return (
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
        <div className="flex items-center gap-3">
          <FileText className="h-6 w-6 text-primary" />
          <CardTitle>{title}</CardTitle>
        </div>
        <Button onClick={handleExport} variant="outline">
          <Download className="mr-2 h-4 w-4" />
          Export as Text
        </Button>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="summary">
          <TabsList className="mb-4">
            <TabsTrigger value="summary">Summary</TabsTrigger>
            {sections.length > 0 && (
              <TabsTrigger value="sections">By Section</TabsTrigger>
            )}
          </TabsList>
          
          <TabsContent value="summary">
            <div className="rounded-lg bg-muted p-6">
              <pre className="whitespace-pre-wrap font-sans text-sm">
                {feedback}
              </pre>
            </div>
          </TabsContent>
          
          {sections.length > 0 && (
            <TabsContent value="sections">
              <div className="space-y-6">
                {sections.map((section, index) => (
                  <div key={index} className="border rounded-lg p-4">
                    <div className="flex justify-between items-center mb-2">
                      <h3 className="font-medium">{section.title}</h3>
                      <span className="text-sm bg-primary/10 text-primary px-2 py-1 rounded-full">
                        {section.score.toFixed(1)}/10
                      </span>
                    </div>
                    <Progress 
                      value={section.score * 10} 
                      className="h-2 mb-3" 
                    />
                    <p className="text-sm text-gray-600">{section.feedback}</p>
                  </div>
                ))}
              </div>
            </TabsContent>
          )}
        </Tabs>
      </CardContent>
      {onGoBack && (
        <CardFooter className="pt-4 pb-6 flex justify-center">
          <Button onClick={onGoBack} className="w-full max-w-xs">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Upload New PDF
          </Button>
        </CardFooter>
      )}
    </Card>
  );
};
