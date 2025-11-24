import { GoogleGenAI, Type } from "@google/genai";
import { AgentRole, LogEntry, ScrapedContent, SWOT } from "../types";
import { MODELS, PRICING } from "../constants";
import { PROMPTS } from "../constants";

// --- Configuration ---
// Strictly usage of process.env.API_KEY as per backend/environment standards.
// We do not use frontend-specific accessors like import.meta here.
const API_KEY = process.env.API_KEY;

const calculateCost = (model: string, usage: { promptTokenCount?: number, candidatesTokenCount?: number } | undefined) => {
  if (!usage) return 0;
  const pricing = PRICING[model as keyof typeof PRICING] || PRICING['gemini-2.5-flash'];
  const inputCost = (usage.promptTokenCount || 0) / 1000 * pricing.input;
  const outputCost = (usage.candidatesTokenCount || 0) / 1000 * pricing.output;
  return inputCost + outputCost;
};

// --- 1. Router Agent ---
export const runRouterAgent = async (query: string): Promise<{ target_company: string, analysis_type: string, search_queries: string[], log: LogEntry }> => {
  const startTime = performance.now();
  const ai = new GoogleGenAI({ apiKey: API_KEY });
  
  try {
    const response = await ai.models.generateContent({
      model: MODELS.ROUTER,
      contents: query,
      config: {
        systemInstruction: PROMPTS.ROUTER_SYSTEM,
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            target_company: { type: Type.STRING },
            analysis_type: { type: Type.STRING },
            search_queries: { type: Type.ARRAY, items: { type: Type.STRING } }
          },
          required: ["target_company", "analysis_type", "search_queries"]
        }
      }
    });

    const result = JSON.parse(response.text || "{}");
    const endTime = performance.now();
    
    return {
      ...result,
      log: {
        timestamp: new Date().toISOString(),
        agent: AgentRole.ROUTER,
        message: `Identified target: ${result.target_company} (${result.analysis_type})`,
        type: 'success',
        latencyMs: endTime - startTime,
        tokenUsage: (response.usageMetadata?.totalTokenCount || 0),
        cost: calculateCost(MODELS.ROUTER, response.usageMetadata)
      }
    };
  } catch (error) {
    console.error("Router Agent Error:", error);
    throw error;
  }
};

// --- 2. Hunter Agent (Using Google Search Grounding) ---
export const runHunterAgent = async (company: string, queries: string[]): Promise<{ discoveredUrls: ScrapedContent[], log: LogEntry }> => {
  const startTime = performance.now();
  const ai = new GoogleGenAI({ apiKey: API_KEY });
  
  // Logic Rewire: Incorporate the Router's specific search queries into the prompt
  // to better guide the model's search tool.
  const queryContext = queries.length > 0 
    ? `Specifically investigate these topics: ${queries.join(", ")}.` 
    : "";

  try {
    const response = await ai.models.generateContent({
      model: MODELS.HUNTER,
      contents: `Find the latest official news, pricing, and feature announcements for ${company}. ${queryContext} Be thorough and prioritize recent data.`,
      config: {
        tools: [{ googleSearch: {} }],
      }
    });

    // Extract grounding chunks to simulate "discovered URLs"
    const groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
    
    let discoveredUrls: ScrapedContent[] = groundingChunks
      .filter((chunk: any) => chunk.web?.uri && chunk.web?.title)
      .map((chunk: any) => ({
        url: chunk.web.uri,
        title: chunk.web.title,
        snippet: (response.text || "").slice(0, 200) + "...", 
        content: response.text || "" // Using generated text as context
      }));

    // Fallback if grounding returns no structured chunks but we have text
    if (discoveredUrls.length === 0 && response.text) {
        discoveredUrls = [{
            url: `https://google.com/search?q=${encodeURIComponent(company)}`,
            title: `${company} Search Results`,
            snippet: response.text.slice(0, 150) + "...",
            content: response.text
        }];
    }

    // Optimize: Limit context passed to next agent to prevent token overload
    const optimizedUrls = discoveredUrls.map(u => ({
        ...u,
        content: u.content.slice(0, 2000) // Truncate large contents
    }));

    const endTime = performance.now();

    return {
      discoveredUrls: optimizedUrls,
      log: {
        timestamp: new Date().toISOString(),
        agent: AgentRole.HUNTER,
        message: `Discovered ${discoveredUrls.length} high-signal URLs via Google Search`,
        type: 'success',
        latencyMs: endTime - startTime,
        tokenUsage: (response.usageMetadata?.totalTokenCount || 0),
        cost: calculateCost(MODELS.HUNTER, response.usageMetadata)
      }
    };
  } catch (error) {
    console.error("Hunter Agent Error:", error);
    throw error;
  }
};

// --- 3. Scraper Agent (Simulated Processing) ---
export const runScraperAgent = async (urls: ScrapedContent[]): Promise<{ extractedContent: string, log: LogEntry }> => {
  const startTime = performance.now();
  const ai = new GoogleGenAI({ apiKey: API_KEY });
  
  // Context Engineering: Truncate and Consolidate
  const rawText = urls.length > 0 
    ? urls.map(u => `Source: ${u.url}\nTitle: ${u.title}\nSummary: ${u.content.slice(0, 2000)}`).join("\n\n")
    : "No specific URLs found, please analyze general knowledge about the company.";

  try {
    const response = await ai.models.generateContent({
      model: MODELS.SCRAPER,
      contents: `You are a Data Engineer. Clean and consolidate the following competitive intelligence data. 
      Remove duplicates, noise, and marketing fluff. Preserve facts, numbers, dates, and pricing.
      Keep the output concise (under 4000 tokens).
      
      RAW DATA:
      ${rawText}`,
    });

    const endTime = performance.now();
    
    // Fallback for empty response
    const content = response.text && response.text.length > 0 
        ? response.text 
        : "Data extracted but no summary generated.";

    return {
      extractedContent: content,
      log: {
        timestamp: new Date().toISOString(),
        agent: AgentRole.SCRAPER,
        message: `Processed and cleaned content from ${urls.length} sources`,
        type: 'success',
        latencyMs: endTime - startTime,
        tokenUsage: (response.usageMetadata?.totalTokenCount || 0),
        cost: calculateCost(MODELS.SCRAPER, response.usageMetadata)
      }
    };
  } catch (error) {
     console.error("Scraper Agent Error:", error);
     throw error;
  }
};

// --- 4. Analyst Agent ---
export const runAnalystAgent = async (content: string): Promise<{ swot: SWOT, log: LogEntry }> => {
  const startTime = performance.now();
  const ai = new GoogleGenAI({ apiKey: API_KEY });

  // Safeguard against empty content
  const safeContent = content && content.trim().length > 10 ? content : "No data available for analysis.";

  try {
    const response = await ai.models.generateContent({
      model: MODELS.ANALYST,
      contents: `Data: ${safeContent}`,
      config: {
        systemInstruction: PROMPTS.ANALYST_SYSTEM,
        responseMimeType: "application/json",
        responseSchema: {
            type: Type.OBJECT,
            properties: {
                strengths: { type: Type.ARRAY, items: { type: Type.STRING } },
                weaknesses: { type: Type.ARRAY, items: { type: Type.STRING } },
                opportunities: { type: Type.ARRAY, items: { type: Type.STRING } },
                threats: { type: Type.ARRAY, items: { type: Type.STRING } },
                scores: {
                  type: Type.OBJECT,
                  properties: {
                    innovation: { type: Type.INTEGER },
                    market_share: { type: Type.INTEGER },
                    pricing_power: { type: Type.INTEGER },
                    brand_reputation: { type: Type.INTEGER },
                    velocity: { type: Type.INTEGER },
                  },
                  required: ["innovation", "market_share", "pricing_power", "brand_reputation", "velocity"]
                }
            },
            required: ["strengths", "weaknesses", "opportunities", "threats", "scores"]
        }
      }
    });

    let swot: SWOT;
    try {
        swot = JSON.parse(response.text || "{}");
        // Ensure scores exist if the model hallucinates a partial object
        if (!swot.scores) {
          swot.scores = {
            innovation: 50,
            market_share: 50,
            pricing_power: 50,
            brand_reputation: 50,
            velocity: 50
          };
        }
    } catch (e) {
        console.warn("JSON parse failed, using fallback empty SWOT", e);
        swot = { 
          strengths: [], weaknesses: [], opportunities: [], threats: [],
          scores: { innovation: 0, market_share: 0, pricing_power: 0, brand_reputation: 0, velocity: 0 }
        };
    }

    const endTime = performance.now();

    return {
      swot,
      log: {
        timestamp: new Date().toISOString(),
        agent: AgentRole.ANALYST,
        message: `Generated SWOT analysis with ${Object.values(swot.scores).reduce((a, b) => a + b, 0) / 5}% avg score`,
        type: 'success',
        latencyMs: endTime - startTime,
        tokenUsage: (response.usageMetadata?.totalTokenCount || 0),
        cost: calculateCost(MODELS.ANALYST, response.usageMetadata)
      }
    };

  } catch (error) {
    console.error("Analyst Agent Error:", error);
    throw error;
  }
};

// --- 5. Reporter Agent ---
export const runReporterAgent = async (swot: SWOT, rawData: string, company: string): Promise<{ report: string, log: LogEntry }> => {
  const startTime = performance.now();
  const ai = new GoogleGenAI({ apiKey: API_KEY });

  const swotText = JSON.stringify(swot, null, 2);
  
  try {
    const response = await ai.models.generateContent({
      model: MODELS.REPORTER,
      contents: `Company: ${company}\n\nSWOT Analysis:\n${swotText}\n\nContext Data:\n${rawData}`,
      config: {
        systemInstruction: PROMPTS.REPORTER_SYSTEM
      }
    });

    const endTime = performance.now();
    
    // Robustness check
    if (!response.text) {
        throw new Error("Reporter agent returned empty content.");
    }

    return {
      report: response.text,
      log: {
        timestamp: new Date().toISOString(),
        agent: AgentRole.REPORTER,
        message: `Finalized executive report for ${company}`,
        type: 'success',
        latencyMs: endTime - startTime,
        tokenUsage: (response.usageMetadata?.totalTokenCount || 0),
        cost: calculateCost(MODELS.REPORTER, response.usageMetadata)
      }
    };
  } catch (error) {
     console.error("Reporter Agent Error:", error);
     
     // Return a graceful failure log instead of crashing app
     return {
        report: "# Report Generation Failed\n\nThe AI model encountered an error generating the final report. Please try again.",
        log: {
            timestamp: new Date().toISOString(),
            agent: AgentRole.REPORTER,
            message: `Failed to generate report: ${error}`,
            type: 'error',
            latencyMs: 0,
            tokenUsage: 0,
            cost: 0
        }
     };
  }
};

// --- 6. Chat Session Helper ---
export const createChatSession = (context: string) => {
  const ai = new GoogleGenAI({ apiKey: API_KEY });
  return ai.chats.create({
    model: MODELS.ANALYST,
    config: {
      tools: [{ googleSearch: {} }], // Enable search so it can answer "anything" even outside context
      systemInstruction: `You are a Senior Competitive Intelligence Analyst.
      
      You have access to a generated report context (provided below), which you should use as a primary source if the user asks about the specific target company.
      
      CRITICAL INSTRUCTION:
      You are NOT limited to the provided context.
      If the user asks about a different company, a general topic, or something not in the report (e.g., asking about TCS when the report is about Tata Motors), you MUST answer using your general knowledge and the Google Search tool.
      
      DO NOT say "The report does not contain information about...".
      DO NOT apologize for missing context.
      Simply answer the question to the best of your ability using all tools available to you.
      
      CONTEXT DATA:
      ${context}
      
      Guidelines:
      - Be concise, professional, and data-driven.
      - Use bullet points for lists.`,
    }
  });
};
