import { Shield } from 'lucide-react';

export default function Navbar({ activePage, onNavigate }) {
  const links = [
    { id: 'home', label: 'Home' },
    { id: 'analyse', label: 'Analyse' },
    { id: 'demo', label: 'Demo' },
  ];

  return (
    <nav className="navbar" role="navigation" aria-label="Main navigation">
      <div className="container navbar-inner">
        <a
          className="navbar-brand"
          href="#"
          id="navbar-brand"
          onClick={(e) => { e.preventDefault(); onNavigate('home'); }}
          aria-label="CrowdRisk home"
        >
          <div className="navbar-brand-icon" aria-hidden="true">
            <Shield size={18} color="white" />
          </div>
          <div>
            <div className="navbar-logo-text">CrowdRisk</div>
            <div className="navbar-logo-sub">AI Safety Research</div>
          </div>
        </a>

        <ul className="nav-links" role="list">
          {links.map((l) => (
            <li key={l.id} role="listitem">
              <button
                id={`nav-${l.id}`}
                className={`nav-link${activePage === l.id ? ' active' : ''}`}
                onClick={() => onNavigate(l.id)}
                aria-current={activePage === l.id ? 'page' : undefined}
              >
                {l.label}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  );
}
