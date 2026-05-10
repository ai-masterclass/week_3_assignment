"use client";

interface StatusLineProps {
  message: string;
  isActive: boolean;
  currentStage?: string;
}

export function StatusLine({
  message,
  isActive,
  currentStage = "",
}: StatusLineProps) {
  const getStageIcon = (stage: string): string => {
    const iconMap: Record<string, string> = {
      connecting: "LINK",
      refining: "REFN",
      refined: "DONE",
      semantic_search: "VSRC",
      semantic_done: "DONE",
      bm25_search: "TSRC",
      bm25_done: "DONE",
      fusion: "FUSE",
      reranking: "RANK",
      found: "DOCS",
      generating: "GEN",
      writing: "STRM",
      complete: "DONE",
      error: "ERR!",
    };
    return iconMap[stage] || "WAIT";
  };

  const stageCode = getStageIcon(currentStage);
  const displayMessage = message || "Processing...";

  return (
    <div className="rounded-lg border border-white/5 bg-[#0d0d14]/80 px-4 py-2.5 glow-border">
      <div className="flex items-center gap-3">
        {/* Stage Code */}
        <span className={`font-mono text-[10px] tracking-wider px-2 py-0.5 rounded ${
          currentStage === 'error' 
            ? 'bg-red-500/10 text-red-400 border border-red-500/20' 
            : currentStage === 'complete'
            ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
            : 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20'
        }`}>
          {stageCode}
        </span>

        {/* Spinner */}
        {isActive && (
          <div className="flex gap-1">
            <span className="w-1 h-1 rounded-full bg-cyan-400 animate-pulse" style={{ animationDelay: '0ms' }} />
            <span className="w-1 h-1 rounded-full bg-cyan-400 animate-pulse" style={{ animationDelay: '150ms' }} />
            <span className="w-1 h-1 rounded-full bg-cyan-400 animate-pulse" style={{ animationDelay: '300ms' }} />
          </div>
        )}

        {/* Message */}
        <p className={`text-xs md:text-sm font-medium truncate flex-1 ${
          isActive ? "text-gray-300" : "text-gray-500"
        }`}>
          {displayMessage}
        </p>

        {/* Active indicator line */}
        {isActive && (
          <div className="w-16 h-0.5 rounded-full bg-gradient-to-r from-cyan-500/50 to-transparent overflow-hidden">
            <div className="w-full h-full bg-cyan-400 animate-pulse" />
          </div>
        )}
      </div>
    </div>
  );
}
