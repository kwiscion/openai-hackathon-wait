import { env } from './env';

// Update to point to our backend
export const API_URL = import.meta.env.VITE_API_URL || env.VITE_API_URL || 'http://localhost:8000';

// Set to false to use the real API by default
export const MOCK_API = (import.meta.env.VITE_MOCK_API === 'true') || (env.VITE_MOCK_API === 'true') || false;

export const mockPDFUploadResponse = {
  status: {
    task_id: 123,
    status: "completed",
    file_name: "mock-paper.pdf",
    current_step: "meta_review",
    history: [
      {
        step: "upload",
        status: "completed",
        message: "File uploaded successfully",
        timestamp: new Date().toISOString()
      },
      {
        step: "gathering_papers",
        status: "completed",
        message: "Found 15 relevant papers for comparison",
        timestamp: new Date().toISOString()
      },
      {
        step: "setting_up_reviews",
        status: "completed",
        message: "Review parameters configured successfully",
        timestamp: new Date().toISOString()
      },
      {
        step: "generating_reviews",
        status: "completed",
        message: "Generated reviews for 5 sections",
        timestamp: new Date().toISOString()
      },
      {
        step: "meta_review",
        status: "completed",
        message: "Final review compiled successfully",
        timestamp: new Date().toISOString()
      }
    ],
    data: {
      title: "Research Paper Analysis",
      feedback: `
Abstract:
The abstract effectively summarizes the main points of the research, but could be more concise. Consider reducing it by 20% while maintaining key findings.

Methodology:
- Well-structured research design
- Clear explanation of data collection methods
- Sample size is appropriate for the study
- Statistical analysis methods are well-justified

Results:
The results are presented clearly with good use of data visualization. However, some graphs could benefit from better labeling.

Discussion:
Strong analysis of findings, but consider addressing the following limitations:
- Potential sampling bias
- External validity considerations
- Impact of time constraints on data collection

References:
All citations follow the required format. Consider including more recent sources (past 3 years) to strengthen the literature review.

Overall Score: 8.5/10
`,
      sections: [
        {
          title: "Abstract",
          score: 7.5,
          feedback: "The abstract effectively summarizes the main points of the research, but could be more concise."
        },
        {
          title: "Methodology",
          score: 9.0,
          feedback: "Well-structured research design with clear explanation of data collection methods."
        },
        {
          title: "Results",
          score: 8.5,
          feedback: "Results are presented clearly with good use of data visualization. Some graphs could benefit from better labeling."
        },
        {
          title: "Discussion",
          score: 8.0,
          feedback: "Strong analysis of findings with some limitations to address."
        },
        {
          title: "References",
          score: 9.5,
          feedback: "All citations follow the required format. Consider including more recent sources."
        }
      ]
    }
  },
  data: {
    title: "Research Paper Analysis",
    feedback: `
Abstract:
The abstract effectively summarizes the main points of the research, but could be more concise. Consider reducing it by 20% while maintaining key findings.

Methodology:
- Well-structured research design
- Clear explanation of data collection methods
- Sample size is appropriate for the study
- Statistical analysis methods are well-justified

Results:
The results are presented clearly with good use of data visualization. However, some graphs could benefit from better labeling.

Discussion:
Strong analysis of findings, but consider addressing the following limitations:
- Potential sampling bias
- External validity considerations
- Impact of time constraints on data collection

References:
All citations follow the required format. Consider including more recent sources (past 3 years) to strengthen the literature review.

Overall Score: 8.5/10
`,
    sections: [
      {
        title: "Abstract",
        score: 7.5,
        feedback: "The abstract effectively summarizes the main points of the research, but could be more concise."
      },
      {
        title: "Methodology",
        score: 9.0,
        feedback: "Well-structured research design with clear explanation of data collection methods."
      },
      {
        title: "Results",
        score: 8.5,
        feedback: "Results are presented clearly with good use of data visualization. Some graphs could benefit from better labeling."
      },
      {
        title: "Discussion",
        score: 8.0,
        feedback: "Strong analysis of findings with some limitations to address."
      },
      {
        title: "References",
        score: 9.5,
        feedback: "All citations follow the required format. Consider including more recent sources."
      }
    ]
  }
}; 