'use client'

import { useEffect, useRef } from 'react'
import { marked } from 'marked'
import hljs from 'highlight.js/lib/core'
import javascript from 'highlight.js/lib/languages/javascript'
import python from 'highlight.js/lib/languages/python'
import bash from 'highlight.js/lib/languages/bash'
import 'highlight.js/styles/github-dark-dimmed.min.css'

hljs.registerLanguage('javascript', javascript)
hljs.registerLanguage('python', python)
hljs.registerLanguage('bash', bash)

interface AnswerSectionProps {
  answer: string
  processingTime?: number
  isVisible: boolean
}

export function AnswerSection({ answer, processingTime, isVisible }: AnswerSectionProps) {
  const contentRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (contentRef.current && answer) {
      const renderMarkdown = async () => {
        try {
          const html = await marked.parse(answer)
          if (contentRef.current) {
            contentRef.current.innerHTML = html as string
            
            setTimeout(() => {
              if (contentRef.current) {
                contentRef.current.querySelectorAll('pre code').forEach((block) => {
                  hljs.highlightElement(block as HTMLElement)
                })
              }
            }, 0)
          }
        } catch (error) {
          console.error('Error parsing markdown:', error)
          if (contentRef.current) {
            contentRef.current.innerHTML = answer
          }
        }
      }
      
      renderMarkdown()
    }
  }, [answer])

  if (!isVisible) return null

  return (
    <div className="gradient-border overflow-hidden fade-in">
      <div className="bg-[#0d0d14] rounded-2xl overflow-hidden">
        {/* Header */}
        <div className="px-4 sm:px-6 md:px-8 py-4 border-b border-white/5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 bg-cyan-400 rounded-full shadow-[0_0_8px_rgba(0,255,200,0.4)]" />
            <h3 className="text-base md:text-lg font-semibold text-gray-200">Answer</h3>
          </div>
          {processingTime && (
            <span className="text-[10px] md:text-xs text-cyan-400/60 font-mono bg-cyan-500/5 px-2.5 py-1 rounded-full border border-cyan-500/10">
              {processingTime.toFixed(1)}s
            </span>
          )}
        </div>

        {/* Content */}
        <div className="px-4 sm:px-6 md:px-8 py-5 sm:py-6 md:py-8">
          <div
            ref={contentRef}
            className="prose prose-invert prose-sm sm:prose-base md:prose-lg max-w-none 
              prose-headings:text-gray-100 prose-headings:font-semibold
              prose-p:text-gray-300 prose-p:leading-relaxed
              prose-strong:text-gray-100 prose-strong:font-semibold
              prose-code:text-cyan-300 prose-code:bg-cyan-500/10 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-code:font-mono
              prose-pre:bg-[#111118] prose-pre:border prose-pre:border-white/5 prose-pre:rounded-xl
              prose-a:text-cyan-400 prose-a:no-underline hover:prose-a:underline hover:prose-a:text-cyan-300
              prose-ul:text-gray-300 prose-ol:text-gray-300
              prose-li:text-gray-300"
          />
        </div>
      </div>
    </div>
  )
}
