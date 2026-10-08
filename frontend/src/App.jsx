import { useState } from 'react';
import './index.css';

import Navbar from './components/Navbar';
import LandingPage from './pages/LandingPage';
import AnalysisPage from './pages/AnalysisPage';

export default function App() {
  // Simple client-side routing via state
  const [page, setPage] = useState('home');

  const navigate = (target) => {
    setPage(target === 'analyse' || target === 'demo' ? 'analyse' : target);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <>
      <Navbar activePage={page} onNavigate={navigate} />

      {page === 'home' && (
        <LandingPage
          onAnalyzeClick={() => navigate('analyse')}
          onDemoClick={() => navigate('demo')}
        />
      )}

      {page === 'analyse' && (
        <AnalysisPage initialMode={page === 'demo' ? 'demo' : 'upload'} />
      )}

      <footer className="footer" role="contentinfo">
        <div className="container footer-inner">
          <p className="footer-copy">
            © 2026 CrowdRisk AI — Research Prototype. All risk scores are non-probabilistic model indices.
          </p>
          <ul className="footer-links" aria-label="Footer links">
            <li><a href="#" className="footer-link" onClick={(e) => e.preventDefault()}>Docs</a></li>
            <li><a href="#" className="footer-link" onClick={(e) => e.preventDefault()}>API</a></li>
            <li>
              <a
                href="https://github.com"
                className="footer-link"
                target="_blank"
                rel="noopener noreferrer"
              >
                GitHub
              </a>
            </li>
          </ul>
        </div>
      </footer>
    </>
  );
}
