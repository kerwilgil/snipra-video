import { useMutation } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import { IconCheck, IconShield } from '../components/icons'
import { Spinner } from '../components/ui'
import { api } from '../lib/api'

/**
 * Pestaña que abre la extension para el emparejamiento de un clic. Vive fuera
 * del shell principal (sin sidebar) porque es una accion puntual, no una
 * seccion de la app -- el usuario llega aca desde chrome.tabs.create(), no
 * navegando. Ver backend/app/api/extension_pairing.py para el flujo completo.
 */
export function PairPage() {
  const [params] = useSearchParams()
  const requestId = params.get('request_id') ?? ''

  const approve = useMutation({
    mutationFn: () => api.approveExtensionPairing(requestId),
  })

  return (
    <div className="app-bg flex min-h-screen items-center justify-center p-6 text-ink">
      <div className="section-card w-full max-w-md text-center">
        <div
          className={`mx-auto flex h-12 w-12 items-center justify-center rounded-full ${
            approve.isSuccess ? 'bg-good/10 text-good' : 'bg-cyan/10 text-cyan'
          }`}
        >
          {approve.isSuccess ? <IconCheck className="h-6 w-6" /> : <IconShield className="h-6 w-6" />}
        </div>

        {!requestId && (
          <>
            <h1 className="mt-4 text-lg font-semibold text-ink">Enlace incompleto</h1>
            <p className="mt-2 text-sm leading-6 text-ink-soft">
              Falta el identificador de la solicitud. Abri este emparejamiento de nuevo desde el
              popup de la extension en vez de esta URL directamente.
            </p>
          </>
        )}

        {requestId && !approve.isSuccess && (
          <>
            <h1 className="mt-4 text-lg font-semibold text-ink">Conectar extension de Snipra</h1>
            <p className="mt-2 text-sm leading-6 text-ink-soft">
              La extension del navegador quiere autenticarse con este Studio para poder crear jobs
              (descargar y procesar videos) sin que copies el token a mano. Aprobalo solo si vos
              iniciaste esta conexion desde el popup de la extension.
            </p>
            {approve.isError && (
              <div className="mt-4 rounded-lg border border-bad/30 bg-bad/[0.06] px-4 py-3 text-left text-sm text-bad">
                {approve.error instanceof Error
                  ? approve.error.message
                  : 'No se pudo aprobar la conexion.'}
              </div>
            )}
            <button
              type="button"
              onClick={() => approve.mutate()}
              disabled={approve.isPending}
              className="btn-primary mt-5 w-full disabled:cursor-not-allowed disabled:opacity-70"
            >
              {approve.isPending ? <Spinner /> : null}
              {approve.isPending ? 'Aprobando...' : 'Aprobar conexion'}
            </button>
          </>
        )}

        {approve.isSuccess && (
          <>
            <h1 className="mt-4 text-lg font-semibold text-ink">Extension conectada</h1>
            <p className="mt-2 text-sm leading-6 text-ink-soft">
              Ya podes cerrar esta pestaña. La extension va a detectar la conexion sola en unos
              segundos.
            </p>
          </>
        )}
      </div>
    </div>
  )
}
