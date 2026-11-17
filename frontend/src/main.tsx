import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode, Suspense } from 'react'
import { createRoot } from 'react-dom/client'
import { RouterProvider } from 'react-router-dom'
import '@fontsource-variable/inter'
import './index.css'
import './styles/tokens.css'
import './styles/base.css'
import './styles/controls.css'
import './styles/shell.css'
import './styles/projects.css'
import './styles/page-layout.css'
import './styles/editor-v2.css'
import { router } from './router.tsx'

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <Suspense fallback={<p className="p-8 text-sm text-ink-mute">Cargando...</p>}>
        <RouterProvider router={router} />
      </Suspense>
    </QueryClientProvider>
  </StrictMode>,
)
