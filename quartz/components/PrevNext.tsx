import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/prevnext.scss"
import { resolveRelative } from "../util/path"
import { classNames } from "../util/lang"

// Previous / next lecture links, read from the note's frontmatter:
//   prev: "[[Random Variables]]"
//   next: "[[Sample Statistics]]"
// Targets are matched by file name (the same rule Obsidian wikilinks use).
function findTarget(name: unknown, allFiles: QuartzComponentProps["allFiles"]) {
  if (typeof name !== "string") return undefined
  const clean = name
    .replace(/^\s*\[\[/, "")
    .replace(/\]\]\s*$/, "")
    .split("|")[0]
    .split("#")[0]
    .trim()
  if (!clean) return undefined
  return allFiles.find((f) => {
    const base = (f.filePath ?? "").split("/").pop()?.replace(/\.md$/, "")
    return base === clean
  })
}

const PrevNext: QuartzComponent = ({ fileData, allFiles, displayClass }: QuartzComponentProps) => {
  const prev = findTarget(fileData.frontmatter?.prev, allFiles)
  const next = findTarget(fileData.frontmatter?.next, allFiles)
  if (!prev && !next) return null
  const here = fileData.slug!
  return (
    <nav class={classNames(displayClass, "prev-next")} aria-label="Lecture navigation">
      <div class="pn-cell pn-prev">
        {prev && (
          <a href={resolveRelative(here, prev.slug!)} class="internal pn-link">
            <span class="pn-label">← Previous</span>
            <span class="pn-title">{prev.frontmatter?.title}</span>
          </a>
        )}
      </div>
      <div class="pn-cell pn-next">
        {next && (
          <a href={resolveRelative(here, next.slug!)} class="internal pn-link">
            <span class="pn-label">Next →</span>
            <span class="pn-title">{next.frontmatter?.title}</span>
          </a>
        )}
      </div>
    </nav>
  )
}

PrevNext.css = style
export default (() => PrevNext) satisfies QuartzComponentConstructor
