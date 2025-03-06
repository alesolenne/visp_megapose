# visp_megapose
Pacchetto wrapper di MegaPose per ROS1

MegaPose è un algoritmo di visione model-based per la stima della posa 6D degli oggetti,
di tipo deep-learning fornito da ViSP: la rete impiegata è stata allenata usando migliaia di
soggetti diversi, non necessita quindi di una fase di re-retraining per nuovi oggetti.
In questo modo è possibile stimare la posa dell’oggetto rispetto alla camera con cui viene ripreso,
la quale fornisce lo stream di immagini necessarie.

Per saperne di più su MegaPose
1) https://megapose6d.github.io/
2) https://visp-doc.inria.fr/doxygen/visp-daily/tutorial-tracking-megapose.html

Per la struttura del pacchetto fare riferimento a: [Wrapper_MegaPose](megapose.pdf)

Lavoro originale per ROS2: https://github.com/m11112089/vision_visp/tree/megapose/visp_megapose
