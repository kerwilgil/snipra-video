import { useState } from 'react'
import { Outlet } from 'react-router-dom'
import { ShellTopbar } from './components/ShellTopbar'
import { Sidebar } from './components/Sidebar'
import { TopbarSlotsContext } from './components/topbar'

function App() {
  const [titleSlot, setTitleSlot] = useState<HTMLDivElement | null>(null)
  const [actionsSlot, setActionsSlot] = useState<HTMLDivElement | null>(null)
  const [searchSlot, setSearchSlot] = useState<HTMLDivElement | null>(null)
  return (
    <TopbarSlotsContext.Provider value={{ titleSlot, actionsSlot, searchSlot }}>
      <div className="app-bg app-shell h-dvh overflow-hidden text-ink">
        <Sidebar />
        <div className="app-workspace">
          <ShellTopbar titleRef={setTitleSlot} actionsRef={setActionsSlot} searchRef={setSearchSlot} />
          <main className="app-main min-h-0 min-w-0 overflow-y-auto">
            <Outlet />
          </main>
        </div>
      </div>
    </TopbarSlotsContext.Provider>
  )
}

export default App
