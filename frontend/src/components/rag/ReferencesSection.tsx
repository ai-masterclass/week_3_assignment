'use client'

interface Source {
  title: string
  authors?: string
  pdf_url?: string
  github_link?: string
  video_link?: string
  acm_url?: string
}

interface ReferencesSectionProps {
  answer: string
  allSources: Source[]
  isVisible: boolean
}

export function ReferencesSection({ answer, allSources, isVisible }: ReferencesSectionProps) {
  const extractCitations = (text: string): string[] => {
    const citationRegex = /\[([^\]]+)\]/g
    const citations = new Set<string>()
    let match
    
    while ((match = citationRegex.exec(text)) !== null) {
      citations.add(match[1])
    }
    
    return Array.from(citations)
  }

  const getCitedSources = (citations: string[], sources: Source[]): Source[] => {
    const citedSources: Source[] = []
    const usedIndices = new Set<number>()

    citations.forEach((citation) => {
      let source = sources.find((s, idx) => {
        if (usedIndices.has(idx)) return false
        return s.title === citation
      })

      if (!source) {
        const citationLower = citation.toLowerCase()
        source = sources.find((s, idx) => {
          if (usedIndices.has(idx)) return false
          return s.title.toLowerCase() === citationLower
        })
      }

      if (!source) {
        const citationLower = citation.toLowerCase()
        source = sources.find((s, idx) => {
          if (usedIndices.has(idx)) return false
          const titleLower = s.title.toLowerCase()
          return citationLower.includes(titleLower) || titleLower.includes(citationLower)
        })
      }

      if (source) {
        const sourceIdx = sources.indexOf(source)
        usedIndices.add(sourceIdx)
        citedSources.push(source)
      }
    })

    return citedSources
  }

  const citations = extractCitations(answer)
  const citedSources = getCitedSources(citations, allSources)

  if (!isVisible || citedSources.length === 0) return null

  const getLinkStyle = (label: string) => {
    const styles: Record<string, string> = {
      PDF: 'text-cyan-400 hover:text-cyan-300 border-cyan-500/20 hover:border-cyan-500/40 hover:bg-cyan-500/5',
      GitHub: 'text-emerald-400 hover:text-emerald-300 border-emerald-500/20 hover:border-emerald-500/40 hover:bg-emerald-500/5',
      Video: 'text-rose-400 hover:text-rose-300 border-rose-500/20 hover:border-rose-500/40 hover:bg-rose-500/5',
      ACM: 'text-purple-400 hover:text-purple-300 border-purple-500/20 hover:border-purple-500/40 hover:bg-purple-500/5',
    }
    return styles[label] || ''
  }

  const getLinks = (source: Source) => {
    const links = []
    if (source.pdf_url) links.push({ url: source.pdf_url, label: 'PDF' })
    if (source.github_link) links.push({ url: source.github_link, label: 'GitHub' })
    if (source.video_link) links.push({ url: source.video_link, label: 'Video' })
    if (source.acm_url) links.push({ url: source.acm_url, label: 'ACM' })
    return links
  }

  return (
    <div className="gradient-border overflow-hidden fade-in">
      <div className="bg-[#0d0d14] rounded-2xl overflow-hidden">
        {/* Header */}
        <div className="px-4 sm:px-6 md:px-8 py-4 border-b border-white/5 flex items-center gap-3">
          <div className="w-2 h-2 bg-purple-400 rounded-full shadow-[0_0_8px_rgba(168,85,247,0.4)]" />
          <h3 className="text-base md:text-lg font-semibold text-gray-200">References</h3>
          <span className="text-[10px] text-purple-400/60 font-mono bg-purple-500/5 px-2 py-0.5 rounded-full border border-purple-500/10">
            {citedSources.length}
          </span>
        </div>

        {/* Mobile: Card layout */}
        <div className="md:hidden divide-y divide-white/[0.03]">
          {citedSources.map((source, index) => {
            const links = getLinks(source)
            return (
              <div key={index} className="px-4 sm:px-6 py-4 hover:bg-white/[0.02] transition-colors fade-in">
                <div className="flex items-start gap-3">
                  <span className="text-gray-600 font-mono text-xs mt-0.5 flex-shrink-0">{index + 1}.</span>
                  <div className="min-w-0 flex-1">
                    <p className="text-gray-200 font-medium text-sm leading-snug">{source.title}</p>
                    {source.authors && (
                      <p className="text-gray-500 text-xs mt-1.5 line-clamp-2">{source.authors}</p>
                    )}
                    {links.length > 0 && (
                      <div className="flex flex-wrap gap-1.5 mt-2.5">
                        {links.map((link, linkIndex) => (
                          <a
                            key={linkIndex}
                            href={link.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className={`${getLinkStyle(link.label)} text-[10px] font-mono font-medium px-2 py-1 rounded border transition-all`}
                          >
                            {link.label}
                          </a>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )
          })}
        </div>

        {/* Desktop: Table layout */}
        <div className="hidden md:block overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-white/5">
                <th className="text-left py-3 px-6 font-mono text-[10px] uppercase tracking-widest text-gray-600">#</th>
                <th className="text-left py-3 px-6 font-mono text-[10px] uppercase tracking-widest text-gray-600">Title</th>
                <th className="text-left py-3 px-6 font-mono text-[10px] uppercase tracking-widest text-gray-600">Authors</th>
                <th className="text-left py-3 px-6 font-mono text-[10px] uppercase tracking-widest text-gray-600">Links</th>
              </tr>
            </thead>
            <tbody>
              {citedSources.map((source, index) => {
                const links = getLinks(source)
                return (
                  <tr 
                    key={index} 
                    className="border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors fade-in"
                  >
                    <td className="py-4 px-6 text-gray-600 font-mono text-sm">{index + 1}</td>
                    <td className="py-4 px-6 text-gray-200 font-medium max-w-md text-sm">
                      {source.title}
                    </td>
                    <td className="py-4 px-6 text-gray-500 text-xs max-w-xs">
                      {source.authors ? (
                        <span title={source.authors} className="line-clamp-2">
                          {source.authors.length > 80 
                            ? `${source.authors.substring(0, 80)}...` 
                            : source.authors}
                        </span>
                      ) : (
                        <span className="text-gray-700">-</span>
                      )}
                    </td>
                    <td className="py-4 px-6">
                      {links.length > 0 ? (
                        <div className="flex flex-wrap gap-1.5">
                          {links.map((link, linkIndex) => (
                            <a
                              key={linkIndex}
                              href={link.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className={`${getLinkStyle(link.label)} text-[10px] font-mono font-medium px-2 py-1 rounded border transition-all`}
                            >
                              {link.label}
                            </a>
                          ))}
                        </div>
                      ) : (
                        <span className="text-gray-700 text-sm">-</span>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
