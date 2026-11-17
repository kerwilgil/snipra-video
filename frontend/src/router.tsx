/* eslint-disable react-refresh/only-export-components -- modulo de configuracion de rutas,
   nunca se hot-reloadea como componente; los lazy() no son el default export del archivo. */
import { lazy } from 'react'
import { createBrowserRouter, Navigate } from 'react-router-dom'
import App from './App.tsx'

// Code-splitting por ruta: cada pagina se baja solo cuando se visita, en vez
// de un unico bundle con Settings/Channels/JobDetail aunque el usuario solo
// abra Upload. Vite separa cada import() dinamico en su propio chunk solo.
const UploadPage = lazy(() => import('./pages/UploadPage.tsx').then((m) => ({ default: m.UploadPage })))
const JobsPage = lazy(() => import('./pages/JobsPage.tsx').then((m) => ({ default: m.JobsPage })))
const JobDetailPage = lazy(() => import('./pages/JobDetailPage.tsx').then((m) => ({ default: m.JobDetailPage })))
const ComposePage = lazy(() => import('./pages/ComposePage.tsx').then((m) => ({ default: m.ComposePage })))
const ChannelsPage = lazy(() => import('./pages/ChannelsPage.tsx').then((m) => ({ default: m.ChannelsPage })))
const SettingsPage = lazy(() => import('./pages/SettingsPage.tsx').then((m) => ({ default: m.SettingsPage })))
const PairPage = lazy(() => import('./pages/PairPage.tsx').then((m) => ({ default: m.PairPage })))

export const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <UploadPage /> },
      { path: 'jobs', element: <JobsPage /> },
      { path: 'jobs/:id', element: <JobDetailPage /> },
      { path: 'editor', element: <Navigate to="/jobs" replace /> },
      { path: 'compose', element: <ComposePage /> },
      { path: 'channels', element: <ChannelsPage /> },
      { path: 'settings', element: <SettingsPage /> },
    ],
  },
  { path: '/editor/*', element: <Navigate to="/jobs" replace /> },
  { path: '/pair', element: <PairPage /> },
])
