import { NavLink } from 'react-router-dom'
import brandDark from '../../image/snipra-brand-v2/logos/snipra-primary-on-dark.png'
import brandLight from '../../image/snipra-brand-v2/logos/snipra-primary-on-light.png'
import brandMonoWhite from '../../image/snipra-brand-v2/logos/snipra-monochrome-white.png'
import appIcon from '../../image/snipra-brand-v2/app-icons/snipra-app-icon-32.png'
import { IconClips, IconRadar, IconScissors, IconSettings, IconUpload } from './icons'

const links = [
  { to: '/', label: 'Subir video', Icon: IconUpload },
  { to: '/jobs', label: 'Proyectos', Icon: IconClips },
  { to: '/compose', label: 'Componer', Icon: IconScissors },
  { to: '/channels', label: 'Canales', Icon: IconRadar },
  { to: '/settings', label: 'Ajustes', Icon: IconSettings },
]

export function Sidebar() {
  return (
    <aside className="app-sidebar">
      <NavLink to="/" className="app-sidebar-brand" title="Snipra">
        <img className="app-sidebar-brand-logo app-sidebar-brand-logo-dark" src={brandDark} alt="Snipra" />
        <img className="app-sidebar-brand-logo app-sidebar-brand-logo-white" src={brandMonoWhite} alt="Snipra" />
        <img className="app-sidebar-brand-logo app-sidebar-brand-logo-light" src={brandLight} alt="Snipra" />
        <img className="app-sidebar-brand-icon" src={appIcon} alt="Snipra" />
      </NavLink>
      <nav className="app-nav" aria-label="Secciones">
        {links.map(({ to, label, Icon }) => (
          <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => (isActive ? 'active' : undefined)}>
            <Icon aria-hidden="true" />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>
      <div className="app-sidebar-footer">
        <div>Creado por Kerwil Gil</div>
        <div>&copy; {new Date().getFullYear()} Snipra. Todos los derechos reservados.</div>
      </div>
    </aside>
  )
}
