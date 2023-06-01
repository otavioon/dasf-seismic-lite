FROM dasf:cpu
COPY dasf-seismic-lite /dasf-seismic-lite
RUN cd /dasf-seismic-lite && pip install .
