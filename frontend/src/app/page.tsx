"use client";

import { QueryForm } from "@/components/rag/QueryForm";
import { AnswerSection } from "@/components/rag/AnswerSection";
import { SourcesSection } from "@/components/rag/SourcesSection";
import { ReferencesSection } from "@/components/rag/ReferencesSection";
import { StatusLine } from "@/components/rag/StatusLine";
import { useRAGStream } from "@/hooks/useRAGStream";

export default function Home() {
  const {
    answer,
    sources,
    processingTime,
    statusMessage,
    currentStage,
    isActive,
    isLoading,
    streamQuery,
    reset,
  } = useRAGStream();

  const handleQuerySubmit = (
    query: string,
    topK: number,
    refineQuery: boolean,
    useReranker: boolean
  ) => {
    reset();
    streamQuery(query, topK, refineQuery, useReranker);
  };

  return (
    <main className="relative min-h-screen flex flex-col z-10">
      <div className="container mx-auto px-4 py-8 md:py-16 max-w-4xl flex-grow flex flex-col">
        <div className="flex-grow">
          {/* Header */}
          <div className="text-center mb-12 md:mb-16 animate-float">
            <div className="inline-block mb-3">
              <span className="text-[10px] md:text-xs uppercase tracking-[0.3em] text-cyan-400/60 font-medium">
                Retrieval-Augmented Generation
              </span>
            </div>
            <h1 className="text-4xl md:text-7xl font-semibold tracking-tight glow-text">
              <span className="bg-gradient-to-r from-cyan-300 via-cyan-400 to-blue-400 bg-clip-text text-transparent">
                SIG-RAG
              </span>
            </h1>
            <p className="text-gray-500 mt-3 text-sm md:text-base tracking-wide">
              AI-powered search for SIGGRAPH 2025 research papers
            </p>
          </div>

          {/* Query Form */}
          <div className="mb-8">
            <QueryForm
              onSubmit={handleQuerySubmit}
              isLoading={isLoading}
              hasAnswer={!!answer}
            />
          </div>

          {/* Status */}
          {(isActive || statusMessage) && (
            <div className="mb-6 fade-in">
              <StatusLine
                message={statusMessage}
                isActive={isActive}
                currentStage={currentStage}
              />
            </div>
          )}

          {/* Answer */}
          <div className="mb-6">
            <AnswerSection
              answer={answer}
              processingTime={processingTime || undefined}
              isVisible={!!answer}
            />
          </div>

          {/* References */}
          <div className="mb-6">
            <ReferencesSection
              answer={answer}
              allSources={sources}
              isVisible={!!answer && sources.length > 0}
            />
          </div>
        </div>

        {/* Footer */}
        <div className="mt-16 pt-8 border-t border-white/5 text-center text-gray-600 text-xs md:text-sm">
          <p className="mb-2">
            Powered by{" "}
            <span className="text-cyan-500/70">
              hooya
            </span>
          </p>
          <p className="flex items-center justify-center gap-4">
            <a
              href="/api/info"
              className="text-gray-600 hover:text-cyan-400 transition-colors"
            >
              API Info
            </a>
            <span className="text-gray-700">|</span>
            <a
              href="/docs"
              className="text-gray-600 hover:text-cyan-400 transition-colors"
            >
              API Docs
            </a>
            <span className="text-gray-700">|</span>
            <a
              href="/health"
              className="text-gray-600 hover:text-cyan-400 transition-colors"
            >
              Health
            </a>
          </p>
        </div>
      </div>
    </main>
  );
}
