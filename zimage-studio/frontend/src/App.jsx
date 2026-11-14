import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

const CATS = [
  { key: 'face', label: 'Khuôn mặt' },
  { key: 'outfit', label: 'Trang phục' },
  { key: 'pose', label: 'Tư thế' },
  { key: 'background', label: 'Bối cảnh' },
]
const SIZES = [
  ['1024x1024', 'Vuông 1024×1024'],
  ['832x1216', 'Dọc 832×1216'],
  ['896x1152', 'Dọc 896×1152'],
  ['1216x832', 'Ngang 1216×832'],
  ['720x1280', 'Dọc 9:16 720×1280'],
]

async function api(path, opts = {}) {
  const res = await fetch(path, opts)
  if (!res.ok) {
    let msg = res.statusText
    try { msg = (await res.json()).detail || msg } catch { /* ignore */ }
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg))
  }
  return res.json()
}
const json = (method, body) => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export default function App() {
  const [tab, setTab] = useState('generate')
  const [library, setLibrary] = useState([])
  const [status, setStatus] = useState(null)
  const [jobs, setJobs] = useState([])
  const [outputs, setOutputs] = useState([])
  const [toast, setToast] = useState(null)
  const [viewer, setViewer] = useState(null)

  const notify = useCallback((msg, err = false) => {
    setToast({ msg, err })
    setTimeout(() => setToast(null), 4000)
  }, [])

  const loadLibrary = useCallback(() => api('/api/library').then(setLibrary).catch((e) => notify(e.message, true)), [notify])
  const loadOutputs = useCallback(() => api('/api/outputs').then(setOutputs).catch(() => {}), [])

  useEffect(() => { loadLibrary(); loadOutputs() }, [loadLibrary, loadOutputs])

  // poll status + jobs
  const doneCount = useRef(0)
  useEffect(() => {
    let alive = true
    const tick = async () => {
      try {
        const [s, j] = await Promise.all([api('/api/status'), api('/api/jobs')])
        if (!alive) return
        setStatus(s)
        setJobs(j)
        const done = j.filter((x) => x.status === 'done').length
        if (done !== doneCount.current) { doneCount.current = done; loadOutputs() }
      } catch {
        if (alive) setStatus({ status: 'không kết nối được backend', error: 'offline' })
      }
    }
    tick()
    const id = setInterval(tick, 1500)
    return () => { alive = false; clearInterval(id) }
  }, [loadOutputs])

  return (
    <div className="app">
      <header>
        <h1>Z-Image Studio</h1>
        <nav>
          {[['generate', 'Tạo ảnh'], ['library', 'Thư viện'], ['gallery', 'Kết quả']].map(([k, l]) => (
            <button key={k} className={tab === k ? 'active' : ''} onClick={() => setTab(k)}>{l}</button>
          ))}
        </nav>
        <StatusBadge status={status} />
      </header>

      <main>
        {tab === 'library' && <Library library={library} reload={loadLibrary} notify={notify} />}
        {tab === 'generate' && (
          <Generate library={library} jobs={jobs} notify={notify} onOpen={setViewer} goLibrary={() => setTab('library')} />
        )}
        {tab === 'gallery' && <Gallery outputs={outputs} reload={loadOutputs} onOpen={setViewer} />}
      </main>

      {viewer && <Viewer item={viewer} onClose={() => setViewer(null)} />}
      {toast && <div className={'toast ' + (toast.err ? 'err' : '')}>{toast.msg}</div>}
    </div>
  )
}

function StatusBadge({ status }) {
  if (!status) return <span className="badge">...</span>
  const ok = status.status === 'sẵn sàng'
  return (
    <span className={'badge ' + (status.error ? 'bad' : ok ? 'good' : 'busy')} title={status.error || ''}>
      {status.status}{status.device ? ` · ${status.device}` : ''}{status.pending ? ` · ${status.pending} đang chờ` : ''}
    </span>
  )
}

/* ------------------------------------------------------------------ Library */

function Library({ library, reload, notify }) {
  const [cat, setCat] = useState('face')
  const [busy, setBusy] = useState({})
  const [skeleton, setSkeleton] = useState(false)
  const fileRef = useRef()
  const items = library.filter((i) => i.category === cat)

  const upload = async (files) => {
    for (const f of files) {
      const fd = new FormData()
      fd.append('file', f)
      fd.append('category', cat)
      fd.append('name', f.name.replace(/\.[^.]+$/, ''))
      fd.append('is_skeleton', cat === 'pose' && skeleton ? 'true' : 'false')
      try { await api('/api/library', { method: 'POST', body: fd }) } catch (e) { notify(e.message, true) }
    }
    reload()
  }

  const run = async (id, label, fn) => {
    setBusy((b) => ({ ...b, [id]: label }))
    try { await fn(); await reload() } catch (e) { notify(e.message, true) }
    setBusy((b) => { const n = { ...b }; delete n[id]; return n })
  }

  return (
    <section>
      <div className="row tabs">
        {CATS.map((c) => (
          <button key={c.key} className={cat === c.key ? 'active' : ''} onClick={() => setCat(c.key)}>
            {c.label} <small>({library.filter((i) => i.category === c.key).length})</small>
          </button>
        ))}
      </div>

      <div
        className="dropzone"
        onClick={() => fileRef.current.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => { e.preventDefault(); upload([...e.dataTransfer.files]) }}
      >
        Kéo thả hoặc bấm để thêm ảnh vào “{CATS.find((c) => c.key === cat).label}”
        <input ref={fileRef} type="file" accept="image/*" multiple hidden onChange={(e) => { upload([...e.target.files]); e.target.value = '' }} />
      </div>
      {cat === 'pose' && (
        <label className="check">
          <input type="checkbox" checked={skeleton} onChange={(e) => setSkeleton(e.target.checked)} />
          Ảnh tải lên đã là skeleton OpenPose (không cần trích)
        </label>
      )}
      <p className="hint">
        {cat === 'pose'
          ? 'Tư thế được áp dụng bằng ControlNet: tải ảnh người thật rồi bấm “Trích tư thế”, hoặc tải thẳng ảnh skeleton.'
          : 'Z-Image-Turbo nhận văn bản, nên mỗi ảnh cần một mô tả (tiếng Anh cho kết quả tốt nhất). Bấm “Tự mô tả” để AI viết giúp, rồi sửa lại nếu cần.'}
      </p>

      <div className="lib-grid">
        {items.map((it) => (
          <LibCard key={it.id} it={it} busy={busy[it.id]} run={run} />
        ))}
        {!items.length && <p className="empty">Chưa có ảnh nào.</p>}
      </div>
    </section>
  )
}

function LibCard({ it, busy, run }) {
  const [name, setName] = useState(it.name)
  const [desc, setDesc] = useState(it.description || '')
  useEffect(() => { setName(it.name); setDesc(it.description || '') }, [it.name, it.description])
  const dirty = name !== it.name || desc !== (it.description || '')

  return (
    <div className="card">
      <div className="thumbs">
        <img src={`/files/library/${it.file}`} alt={it.name} />
        {it.pose_file && it.pose_file !== it.file && <img src={`/files/library/${it.pose_file}`} alt="pose" />}
      </div>
      <input value={name} onChange={(e) => setName(e.target.value)} />
      <textarea rows={4} placeholder="Mô tả (vd: young woman, oval face, long black hair...)" value={desc} onChange={(e) => setDesc(e.target.value)} />
      <div className="row wrap">
        {dirty && (
          <button className="primary" disabled={!!busy}
            onClick={() => run(it.id, 'Đang lưu', () => api(`/api/library/${it.id}`, json('PATCH', { name, description: desc })))}>
            Lưu
          </button>
        )}
        <button disabled={!!busy} onClick={() => run(it.id, 'Đang mô tả...', () => api(`/api/library/${it.id}/describe`, { method: 'POST' }))}>
          Tự mô tả
        </button>
        {it.category === 'pose' && (
          <button disabled={!!busy} onClick={() => run(it.id, 'Đang trích...', () => api(`/api/library/${it.id}/extract-pose`, { method: 'POST' }))}>
            Trích tư thế
          </button>
        )}
        <button className="danger" disabled={!!busy}
          onClick={() => confirm('Xoá ảnh này?') && run(it.id, 'Đang xoá', () => api(`/api/library/${it.id}`, { method: 'DELETE' }))}>
          Xoá
        </button>
      </div>
      {busy && <div className="busy">{busy}</div>}
      {it.category === 'pose' && !it.pose_file && <div className="warn">Chưa có skeleton</div>}
    </div>
  )
}

/* ------------------------------------------------------------------ Generate */

function Generate({ library, jobs, notify, onOpen, goLibrary }) {
  const [sel, setSel] = useState({ face: [], outfit: [], pose: [], background: [] })
  const [prompt, setPrompt] = useState('A high quality photo of a person, natural lighting, sharp focus, 85mm')
  const [size, setSize] = useState('832x1216')
  const [count, setCount] = useState(2)
  const [steps, setSteps] = useState(9)
  const [seed, setSeed] = useState(-1)
  const [cscale, setCscale] = useState(0.75)
  const [usePose, setUsePose] = useState(true)
  const [sending, setSending] = useState(false)

  const toggle = (cat, id) => setSel((s) => ({
    ...s, [cat]: s[cat].includes(id) ? s[cat].filter((x) => x !== id) : [...s[cat], id],
  }))
  const combos = CATS.reduce((n, c) => n * Math.max(1, sel[c.key].length), 1)
  const total = combos * count

  const preview = useMemo(() => {
    const first = (cat) => library.find((i) => i.id === sel[cat][0])
    const parts = { face: first('face'), outfit: first('outfit'), pose: first('pose'), background: first('background') }
    const d = (k) => (parts[k]?.description || '').trim()
    if (CATS.some((c) => prompt.includes(`{${c.key}}`))) {
      return CATS.reduce((p, c) => p.replaceAll(`{${c.key}}`, d(c.key)), prompt)
    }
    const labels = { face: 'The person', outfit: 'Wearing', pose: 'Pose', background: 'Setting' }
    return [prompt.trim(), ...CATS.filter((c) => d(c.key)).map((c) => `${labels[c.key]}: ${d(c.key)}`)]
      .filter(Boolean).map((s) => s.replace(/[.\s]+$/, '')).join('. ') + '.'
  }, [library, sel, prompt])

  const submit = async () => {
    const [width, height] = size.split('x').map(Number)
    setSending(true)
    try {
      const r = await api('/api/generate', json('POST', {
        prompt, width, height, steps, seed: Number(seed), images_per_combo: count,
        control_scale: cscale, use_pose_control: usePose,
        face_ids: sel.face, outfit_ids: sel.outfit, pose_ids: sel.pose, background_ids: sel.background,
      }))
      notify(`Đã xếp hàng ${r.jobs.length} ảnh`)
    } catch (e) { notify(e.message, true) }
    setSending(false)
  }

  const recent = jobs.slice(0, 60)

  return (
    <div className="gen">
      <section className="panel">
        {CATS.map((c) => {
          const items = library.filter((i) => i.category === c.key)
          return (
            <div key={c.key} className="pick">
              <h3>{c.label} <small>{sel[c.key].length ? `${sel[c.key].length} đã chọn` : 'không dùng'}</small></h3>
              <div className="pick-row">
                {items.map((it) => (
                  <button key={it.id} className={'pick-item ' + (sel[c.key].includes(it.id) ? 'on' : '')}
                    title={it.description || it.name} onClick={() => toggle(c.key, it.id)}>
                    <img src={`/files/library/${it.file}`} alt={it.name} />
                    <span>{it.name}</span>
                    {c.key !== 'pose' && !it.description && <em>chưa mô tả</em>}
                    {c.key === 'pose' && !it.pose_file && <em>chưa có skeleton</em>}
                  </button>
                ))}
                {!items.length && <button className="link" onClick={goLibrary}>+ Thêm trong Thư viện</button>}
              </div>
            </div>
          )
        })}

        <label>Prompt chính <small>(có thể dùng {'{face} {outfit} {pose} {background}'})</small></label>
        <textarea rows={3} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
        <details>
          <summary>Xem prompt sẽ gửi (tổ hợp đầu tiên)</summary>
          <p className="mono">{preview}</p>
        </details>

        <div className="settings">
          <label>Kích thước
            <select value={size} onChange={(e) => setSize(e.target.value)}>
              {SIZES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </label>
          <label>Ảnh / tổ hợp
            <input type="number" min={1} max={16} value={count} onChange={(e) => setCount(Math.max(1, Math.min(16, +e.target.value || 1)))} />
          </label>
          <label>Steps
            <input type="number" min={1} max={50} value={steps} onChange={(e) => setSteps(+e.target.value || 9)} />
          </label>
          <label>Seed (-1 = ngẫu nhiên)
            <input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} />
          </label>
          <label>Độ mạnh pose: {cscale.toFixed(2)}
            <input type="range" min={0} max={1.5} step={0.05} value={cscale} onChange={(e) => setCscale(+e.target.value)} />
          </label>
          <label className="check">
            <input type="checkbox" checked={usePose} onChange={(e) => setUsePose(e.target.checked)} /> Dùng ControlNet pose
          </label>
        </div>

        <button className="primary big" disabled={sending || total > 64} onClick={submit}>
          {sending ? 'Đang gửi...' : `Tạo ${total} ảnh (${combos} tổ hợp × ${count})`}
        </button>
        {total > 64 && <p className="warn">Tối đa 64 ảnh mỗi lần.</p>}
        <button className="ghost" onClick={() => api('/api/jobs/cancel', { method: 'POST' }).then((r) => notify(`Đã huỷ ${r.cancelled} job`))}>
          Huỷ các job đang chờ
        </button>
      </section>

      <section className="results">
        <h3>Hàng đợi & kết quả mới</h3>
        <div className="out-grid">
          {recent.map((j) => (
            <div key={j.id} className={'job ' + j.status}>
              {j.status === 'done'
                ? <img src={`/files/outputs/${j.output}`} alt="" onClick={() => onOpen({ ...j, file: j.output })} />
                : <div className="ph">{({ queued: 'Đang chờ', running: 'Đang tạo...', error: 'Lỗi', cancelled: 'Đã huỷ' })[j.status]}</div>}
              <div className="meta">
                {Object.values(j.labels || {}).join(' · ') || 'chỉ prompt'} · seed {j.seed}
                {j.error && <div className="warn">{j.error}</div>}
              </div>
            </div>
          ))}
          {!recent.length && <p className="empty">Chưa có job nào.</p>}
        </div>
      </section>
    </div>
  )
}

/* ------------------------------------------------------------------ Gallery */

function Gallery({ outputs, reload, onOpen }) {
  return (
    <section>
      <div className="out-grid wide">
        {outputs.map((o) => (
          <div key={o.file} className="job done">
            <img src={`/files/outputs/${o.file}`} alt="" onClick={() => onOpen(o)} />
            <div className="meta">
              {Object.values(o.labels || {}).join(' · ') || 'chỉ prompt'} · seed {o.seed}
              <div className="row">
                <a href={`/files/outputs/${o.file}`} download>Tải</a>
                <button className="link danger" onClick={() => confirm('Xoá ảnh?') && api(`/api/outputs/${o.file}`, { method: 'DELETE' }).then(reload)}>Xoá</button>
              </div>
            </div>
          </div>
        ))}
        {!outputs.length && <p className="empty">Chưa có ảnh nào.</p>}
      </div>
    </section>
  )
}

function Viewer({ item, onClose }) {
  useEffect(() => {
    const k = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [onClose])
  return (
    <div className="viewer" onClick={onClose}>
      <div className="viewer-body" onClick={(e) => e.stopPropagation()}>
        <img src={`/files/outputs/${item.file}`} alt="" />
        <div className="viewer-info">
          <p><b>Seed:</b> {item.seed} · {item.width}×{item.height} · {item.steps} steps</p>
          <p className="mono">{item.prompt}</p>
          <div className="row">
            <button onClick={() => navigator.clipboard?.writeText(item.prompt)}>Copy prompt</button>
            <a className="btn" href={`/files/outputs/${item.file}`} download>Tải xuống</a>
            <button onClick={onClose}>Đóng</button>
          </div>
        </div>
      </div>
    </div>
  )
}
