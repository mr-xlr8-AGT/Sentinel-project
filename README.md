# ⿗ S E N T I N E L

[**Deployed App**](https://sentinel-55600892774.us-west1.run.app/)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![React](https://img.shields.io/badge/react-v19.0-61DAFB?logo=react)
![TypeScript](https://img.shields.io/badge/typescript-v5.0-3178C6?logo=typescript)

> **Autonomous Competitive Intelligence Platform**
>
> *Stop Googling. Start Strategizing.*

---

## ⚡ The Mission

**Sentinel** is not a search engine. It is an **Agentic AI Orchestrator** designed to automate the workflow of a top-tier strategy consultant. 

By chaining specialized AI agents, Sentinel autonomously hunts for real-time market data, scrapes high-signal intelligence, performs deep strategic reasoning (SWOT), and quantifies the abstract—turning messy web data into clear, scored metrics like **Innovation**, **Velocity**, and **Market Share**.

---

## 💡 Problem Statement

Organizations face significant challenges in maintaining competitive awareness:

| Challenge | Impact |
| :--- | :--- |
| 📊 **Manual Research** | Analysts spend 20+ hours weekly on competitive research |
| 💰 **Resource Inefficiency** | Senior analysts dedicate 40% of time to data collection |
| 🔄 **Inconsistent Analysis** | Human bias and fatigue result in missed strategic signals |
| 📉 **Delayed Insights** | Research cycles require 3-5 business days to complete |
| ⚡ **Scalability Constraints** | Monitoring 10+ competitors simultaneously proves impractical |

### Solution
Sentinel addresses these challenges through autonomous AI agents that operate continuously, maintain consistency, and scale infinitely - delivering comprehensive competitive intelligence in real-time.

---

## 🚀 Business Impact

### Operational Efficiency
*   ✅ **95% reduction** in research time (days to minutes)
*   ✅ **60% decrease** in competitive analysis costs
*   ✅ **Infinite scalability** without proportional resource increases

### Strategic Advantages
*   ✅ **Real-time alerts** on competitor moves (pricing, features, campaigns)
*   ✅ **Elimination of human bias** for consistent, data-driven insights
*   ✅ **AI-generated strategic recommendations** for immediate action

### Risk Mitigation
*   ✅ **Early detection** of competitive threats
*   ✅ **Comprehensive market signal capture**
*   ✅ **Continuous monitoring** without coverage gaps

---

## 🧠 The Agentic Chain

Sentinel employs a multi-step, state-machine architecture powered by Google's **Gemini 2.5 Flash** (for speed) and **Gemini 3.0 Pro** (for reasoning).

1.  **⚡ Router Agent** (`gemini-2.5-flash`)
    *   **Role:** The Traffic Controller.
    *   **Task:** Analyzes user intent, extracts target companies, and generates optimized search vectors.

2.  **🔍 Hunter Agent** (`gemini-2.5-flash` + **Google Search**)
    *   **Role:** The Investigator.
    *   **Task:** Executes live queries to find pricing pages, changelogs, and press releases. Filters for high-authority domains.

3.  **🕷️ Scraper Agent** (`gemini-2.5-flash`)
    *   **Role:** The Extractor.
    *   **Task:** Ingests raw noise, ads, and HTML; synthesizes a dense "Fact Sheet" of pure signal.

4.  **🧠 Analyst Agent** (`gemini-3-pro-preview`)
    *   **Role:** The Strategist.
    *   **Task:** Reads the Fact Sheet. Performs a 360° SWOT analysis. **Crucially**, it assigns integer scores (0-100) to abstract concepts (Brand Power, Pricing Power) based on evidence.

5.  **📝 Reporter Agent** (`gemini-2.5-flash`)
    *   **Role:** The Closer.
    *   **Task:** Compiles everything into a C-Level Executive Markdown report.

---

## 🏗️ How Sentinel Implements Architectural Requirements

1. **Multi-Agent System**
   - Implemented as a Sequential Agent Chain via a state machine in `App.tsx`'s `handleAnalyze` including:
     - **Router Agent:** Classifies intent.
     - **Hunter Agent:** Finds data.
     - **Scraper Agent:** Context compaction.
     - **Analyst Agent:** Strategic reasoning.
     - **Reporter Agent:** Generates output.
   - **Sample:**
     ```tsx
     // The Orchestrator
     const handleAnalyze = async () => {
       // ...setup state...
       setState(prev => ({ ...prev, currentAgent: AgentRole.ROUTER }));
       const routerResult = await runRouterAgent(query);
       setState(prev => ({ ...prev, currentAgent: AgentRole.HUNTER }));
       const hunterResult = await runHunterAgent(routerResult.target_company, routerResult.search_queries);
       setState(prev => ({ ...prev, currentAgent: AgentRole.SCRAPER }));
       const scraperResult = await runScraperAgent(hunterResult.discoveredUrls);
       // ...continues to Analyst and Reporter...
     }
     ```

2. **Tools (Built-in Gemini/Google Search Tool)**
   - Integrated via Gemini SDK in `geminiService.ts` (`runHunterAgent`, `createChatSession`).
   - Real-world search grounding enabled by `tools: [{ googleSearch: {} }]` prop.
   - **Sample:**
     ```typescript
     export const runHunterAgent = async (company: string, queries: string[]): Promise<{ discoveredUrls: ScrapedContent[], log: LogEntry }> => {
       const response = await ai.models.generateContent({
         model: MODELS.HUNTER,
         contents: `Find the latest official news...`,
         config: {
           tools: [{ googleSearch: {} }],
         }
       });
       // ...extract groundingMetadata...
     }
     ```

3. **Sessions & Memory**
   - Short-term handled by React state (useState) in `App.tsx`.
   - Long-term memory (history) serialized in browser `localStorage` and loaded on mount.
   - **Sample:**
     ```tsx
     useEffect(() => {
       const savedHistory = localStorage.getItem('sentinel_history');
       if (savedHistory) setHistory(JSON.parse(savedHistory));
     }, []);
     useEffect(() => {
       localStorage.setItem('sentinel_history', JSON.stringify(history));
     }, [history]);
     ```

4. **Context Compaction**
   - Scraper Agent reduces token/irrelevant data before analyst agent gets context (`geminiService.ts`).
   - **Sample:**
     ```typescript
     export const runScraperAgent = async (urls: ScrapedContent[]): Promise<{ extractedContent: string, log: LogEntry }> => {
       const response = await ai.models.generateContent({
         model: MODELS.SCRAPER,
         contents: `You are a Data Engineer. Clean and consolidate...\nRAW DATA:\n${rawText}`
       });
       // ...extract...
     }
     ```

5. **Observability (Logging, Tracing, Metrics)**
   - Custom logging/telemetry with execution time and cost metrics; visualized in sidebar (`App.tsx`, `Sidebar.tsx`).
   - **Sample:**
     ```typescript
     const calculateCost = (model: string, usage: { promptTokenCount?: number, candidatesTokenCount?: number } | undefined) => {
       if (!usage) return 0;
       const pricing = PRICING[model as keyof typeof PRICING];
       // ...
     };
     // ...logging/telemetry with cost & performance...
     ```

6. **A2A Protocol (Agent-to-Agent Communication)**
   - Agents communicate via strictly typed JSON output enforced by schema in Gemini (`geminiService.ts`).
   - **Sample:**
     ```typescript
     // Router Agent Output Schema
     responseSchema: {
       type: Type.OBJECT,
       properties: {
         target_company: { type: Type.STRING },
         analysis_type: { type: Type.STRING },
         search_queries: { type: Type.ARRAY, items: { type: Type.STRING } }
       },
       required: ["target_company", "analysis_type", "search_queries"]
     }
     const routerResult = JSON.parse(response.text || "{}");
     // routerResult.target_company is now passed to runHunterAgent
     ```

---

## 💎 Key Features

### 📊 Visual Intelligence Dashboard
Don't just read the report—see the strategy.
*   **Radar Charts:** Instant visual fingerprint of a company's strategic balance.
*   **HUD Metrics:** High-contrast scoring cards for rapid decision-making.
*   **SWOT Grid:** 2x2 matrix layout for scanning Strengths vs. Threats.

### ⚔️ Battle Mode (Comparison View)
Select any two reports from history to trigger **Head-to-Head**.
*   **Dual-Layer Radar:** Overlays competitor strategies to reveal gaps.
*   **Tale of the Tape:** Direct metric-vs-metric comparison bars.
*   **AI Verdict:** Automatically declares an advantage leader based on the data.

### 💬 Context-Aware Analyst Chat
Talk to your data.
*   **RAG + Web:** The chat knows the generated report context *and* has access to live Google Search.
*   **Deep Reasoning:** Ask follow-up questions: *"Okay, but how does their pricing compare to Tesla?"*

---

## 🛠️ Tech Stack

*   **Core:** React 19, TypeScript, Vite.
*   **AI:** `@google/genai` SDK.
*   **Styling:** Tailwind CSS (Custom "Dark Sentinel" Theme).
*   **Visualization:** Custom SVG Radar Charts.
*   **Rendering:** React Markdown + GFM.

---

## 🚀 Getting Started

### Prerequisites
*   Node.js 18+
*   A Google Cloud Project with Gemini API enabled.

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/mr-xlr8-AGT/sentinel.git
    cd sentinel
    ```

2.  **Install dependencies**
    ```bash
    npm install
    ```

3.  **Configure Environment**
    Create a `.env` file in the root:
    ```env
    API_KEY=your_google_gemini_api_key_here
    ```

4.  **Run the Operator Console**
    ```bash
    npm run dev
    ```

---

## 📸 Usage

1.  **Initialize:** Type a target (e.g., *"Analyze Tata Motors EV Strategy"*).
2.  **Observe:** Watch the Agent Visualizer step through the workflow in real-time.
3.  **Review:** Analyze the scored metrics and SWOT grid.
4.  **Compare:** Go to History, select two rivals, and click **Compare**.

---

*Created By Aditya Gaurav*
