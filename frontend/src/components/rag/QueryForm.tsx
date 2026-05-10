'use client'

import { useState, FormEvent } from 'react'

interface QueryFormProps {
  onSubmit: (query: string, topK: number, refineQuery: boolean, useReranker: boolean) => void
  isLoading: boolean
  hasAnswer?: boolean
}

export function QueryForm({ onSubmit, isLoading, hasAnswer }: QueryFormProps) {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(8)
  const [refineQuery, setRefineQuery] = useState(true)
  const [useReranker, setUseReranker] = useState(true)
  const [showSourcesDropdown, setShowSourcesDropdown] = useState(false)
  const [hoveredTooltip, setHoveredTooltip] = useState<string | null>(null)

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    if (!query.trim() || isLoading) return
    onSubmit(query, topK, refineQuery, useReranker)
  }

  const setQueryFromExample = (exampleQuery: string) => {
    setQuery(exampleQuery)
  }

  const exampleQueries = [
    'What is 3D Gaussian Splatting?',
    'How does cloth simulation work in computer graphics?',
    'What are the latest advances in neural rendering?',
    'Explain fluid simulation techniques',
    'Neural radiance fields applications',
    'Real-time rendering optimization methods'
  ]

  return (
    <div className="w-full">
      <form onSubmit={handleSubmit} className="relative">
        <div className={`relative rounded-2xl border transition-all duration-300 ${
          isLoading 
            ? 'border-cyan-500/30 glow-border-active bg-[#0d0d14]' 
            : 'border-white/10 hover:border-cyan-500/20 glow-border bg-[#0d0d14]/80 hover:bg-[#0d0d14]'
        }`}>
          {/* Search row: input + submit */}
          <div className="flex items-center gap-2 px-3 sm:px-4 py-3 sm:py-4 md:py-5">
            <div className="flex-1 min-w-0">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask anything about SIGGRAPH papers..."
                disabled={isLoading}
                className="w-full bg-transparent text-gray-100 px-1 sm:px-3 py-1 text-sm sm:text-base md:text-lg placeholder:text-gray-600 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed caret-cyan-400"
              />
            </div>

            <button
              type="submit"
              disabled={isLoading || !query.trim()}
              className={`p-2 sm:p-2.5 rounded-xl transition-all duration-200 flex-shrink-0 disabled:opacity-30 disabled:cursor-not-allowed ${
                isLoading
                  ? 'bg-cyan-500/20 text-cyan-400'
                  : 'bg-cyan-500 hover:bg-cyan-400 text-[#0a0a0f] shadow-[0_0_15px_rgba(0,255,200,0.3)] hover:shadow-[0_0_25px_rgba(0,255,200,0.5)]'
              }`}
              title="Search"
            >
              {isLoading ? (
                <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              )}
            </button>
          </div>

          {/* Options bar: below input on all screens */}
          <div className="flex items-center gap-2 px-3 sm:px-4 pb-3 sm:pb-4">
            <div className={`flex items-center gap-1 sm:gap-1.5 px-1.5 sm:px-2 py-1 sm:py-1.5 rounded-lg border transition-colors ${
              refineQuery || useReranker 
                ? 'border-cyan-500/20 bg-cyan-500/5' 
                : 'border-white/5 bg-white/[0.02]'
            }`}>
              {/* Refine Query */}
              <div className="relative">
                <button
                  type="button"
                  onClick={() => setRefineQuery(!refineQuery)}
                  onMouseEnter={() => setHoveredTooltip('refine')}
                  onMouseLeave={() => setHoveredTooltip(null)}
                  className={`p-1.5 rounded-md transition-all flex items-center gap-1.5 ${
                    refineQuery
                      ? 'bg-cyan-500/15 hover:bg-cyan-500/25 shadow-[0_0_10px_rgba(0,255,200,0.1)]'
                      : 'hover:bg-white/5'
                  }`}
                >
                  <svg className={`w-3.5 h-3.5 sm:w-4 sm:h-4 ${refineQuery ? 'text-cyan-400' : 'text-gray-500'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                  </svg>
                  <span className={`text-[10px] sm:text-xs font-medium hidden xs:inline ${refineQuery ? 'text-cyan-400' : 'text-gray-500'}`}>Refine</span>
                </button>
                {hoveredTooltip === 'refine' && (
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 border border-white/10 text-white text-xs rounded-lg shadow-lg whitespace-nowrap z-30 hidden sm:block">
                    <div className="font-semibold mb-1 text-cyan-400">Refine Query</div>
                    <div className="text-gray-400">Expands and optimizes your query for better results</div>
                    <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1">
                      <div className="w-2 h-2 bg-gray-900 border-b border-r border-white/10 rotate-45"></div>
                    </div>
                  </div>
                )}
              </div>
              
              {/* Reranker */}
              <div className="relative">
                <button
                  type="button"
                  onClick={() => setUseReranker(!useReranker)}
                  onMouseEnter={() => setHoveredTooltip('reranker')}
                  onMouseLeave={() => setHoveredTooltip(null)}
                  className={`p-1.5 rounded-md transition-all flex items-center gap-1.5 ${
                    useReranker
                      ? 'bg-cyan-500/15 hover:bg-cyan-500/25 shadow-[0_0_10px_rgba(0,255,200,0.1)]'
                      : 'hover:bg-white/5'
                  }`}
                >
                  <svg className={`w-3.5 h-3.5 sm:w-4 sm:h-4 ${useReranker ? 'text-cyan-400' : 'text-gray-500'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4h13M3 8h9m-9 4h6m4 0l4-4m0 0l4 4m-4-4v12" />
                  </svg>
                  <span className={`text-[10px] sm:text-xs font-medium hidden xs:inline ${useReranker ? 'text-cyan-400' : 'text-gray-500'}`}>Rerank</span>
                </button>
                {hoveredTooltip === 'reranker' && (
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 border border-white/10 text-white text-xs rounded-lg shadow-lg whitespace-nowrap z-30 hidden sm:block">
                    <div className="font-semibold mb-1 text-cyan-400">Reranker</div>
                    <div className="text-gray-400">Re-ranks results for better relevance</div>
                    <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1">
                      <div className="w-2 h-2 bg-gray-900 border-b border-r border-white/10 rotate-45"></div>
                    </div>
                  </div>
                )}
              </div>
              
              {/* Sources */}
              <div className="relative">
                <button
                  type="button"
                  onClick={() => setShowSourcesDropdown(!showSourcesDropdown)}
                  onMouseEnter={() => setHoveredTooltip('sources')}
                  onMouseLeave={() => setHoveredTooltip(null)}
                  className="p-1.5 rounded-md hover:bg-white/5 transition-all relative flex items-center gap-1.5"
                >
                  <svg className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                  </svg>
                  <span className="w-4 h-4 bg-cyan-500 text-[#0a0a0f] text-[10px] font-bold rounded-full flex items-center justify-center shadow-[0_0_8px_rgba(0,255,200,0.3)]">
                    {topK}
                  </span>
                </button>
                {hoveredTooltip === 'sources' && !showSourcesDropdown && (
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 border border-white/10 text-white text-xs rounded-lg shadow-lg whitespace-nowrap z-30 hidden sm:block">
                    <div className="font-semibold mb-1 text-cyan-400">Sources</div>
                    <div className="text-gray-400">Papers to retrieve (currently {topK})</div>
                    <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1">
                      <div className="w-2 h-2 bg-gray-900 border-b border-r border-white/10 rotate-45"></div>
                    </div>
                  </div>
                )}
                
                {showSourcesDropdown && (
                  <>
                    <div 
                      className="fixed inset-0 z-10" 
                      onClick={() => setShowSourcesDropdown(false)}
                    />
                    <div className="absolute top-full left-0 mt-1 bg-[#111118] rounded-lg shadow-lg border border-white/10 py-1 min-w-[100px] z-20">
                      {[5, 8, 10, 15].map((num) => (
                        <button
                          key={num}
                          type="button"
                          onClick={() => {
                            setTopK(num)
                            setShowSourcesDropdown(false)
                          }}
                          className={`w-full text-left px-4 py-2 text-sm transition-colors flex items-center justify-between ${
                            topK === num
                              ? 'bg-cyan-500/10 text-cyan-400 font-medium'
                              : 'text-gray-400 hover:bg-white/5 hover:text-gray-200'
                          }`}
                        >
                          <span>{num}</span>
                          {topK === num && (
                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                          )}
                        </button>
                      ))}
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Suggested Queries */}
          {!hasAnswer && !isLoading && (
            <div className="border-t border-white/5 px-3 sm:px-4 py-2 sm:py-3">
              <div className="space-y-0.5">
                {exampleQueries.slice(0, 6).map((example, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => setQueryFromExample(example)}
                    className="w-full flex items-center gap-2 sm:gap-3 px-2 sm:px-3 py-1.5 sm:py-2 rounded-lg hover:bg-white/[0.03] transition-all text-left group"
                  >
                    <svg className="w-3 h-3 sm:w-3.5 sm:h-3.5 text-gray-700 group-hover:text-cyan-500/50 flex-shrink-0 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    <span className="text-xs sm:text-sm text-gray-600 group-hover:text-gray-400 flex-1 truncate transition-colors">
                      {example}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </form>
    </div>
  )
}
