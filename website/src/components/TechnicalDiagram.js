import styles from './TechnicalDiagram.module.css';

// Technical Diagram Component for displaying technical diagrams in the book
const TechnicalDiagram = ({ src, alt, caption, width = "100%" }) => {
  return (
    <div className={styles.technicalDiagram}>
      <figure>
        <img
          src={src}
          alt={alt}
          style={{ maxWidth: width }}
          className={styles.diagramImage}
        />
        {caption && (
          <figcaption className={styles.diagramCaption}>
            {caption}
          </figcaption>
        )}
      </figure>
    </div>
  );
};

export default TechnicalDiagram;