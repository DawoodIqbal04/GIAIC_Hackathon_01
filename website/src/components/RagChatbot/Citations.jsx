import React from 'react';
import './Citations.css';

const Citations = ({ citations = [] }) => {
  if (!citations || citations.length === 0) {
    return null;
  }

  return (
    <div className="citations-container">
      <h4>Sources:</h4>
      <ul className="citations-list">
        {citations.map((citation, index) => (
          <li key={index} className="citation-item">
            <div className="citation-reference">
              <strong>{citation.source_reference}</strong>
            </div>
            <div className="citation-snippet">
              "{citation.content_snippet}"
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Citations;