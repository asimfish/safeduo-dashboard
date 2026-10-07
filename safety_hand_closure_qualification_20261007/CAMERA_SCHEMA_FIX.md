# Render-only schema failure and correction

The first reconstruction aborted before any image at KeyError beam700:pose. The paired data schema stores e6:beam700:pose and e6:beam700:velocity rather than a global object tensor. runtime_camera_v2 assembles the two native rows from those existing keys. The original source, registration, log and failure.txt remain archived. Camera offset and reconstruction tolerances are unchanged. No native physical trajectory, criterion or original image is changed or rerun.
