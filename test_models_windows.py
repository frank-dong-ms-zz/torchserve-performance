import sys
from ts.metrics.metrics_store import MetricsStore
from ts.torch_handler.base_handler import BaseHandler
from uuid import uuid4
from pprint import pprint

class ModelContext:
    def __init__(self):
        self.manifest = {
            'model': {
                'modelName': 'ptclassifier',
                'serializedFile': 'traced_pt_classifer.pt',
                'modelFile': 'model_ph.py'
            }
        }
        self.system_properties = {
            'model_dir': '<ADD COMPLETE PATH HERE>\share_folder\\model-store\\ptclassifier'
        }

        self.explain = False
        self.metrics = MetricsStore(uuid4(), self.manifest['model']['modelName'])

    def get_request_header(self, idx, exp):
        if exp == 'explain':
            return self.explain
        return False

def main():
    if sys.argv[1] == 'fast':
        from ptclassifier.TransformerSeqClassificationHandler import TransformersSeqClassifierHandler as Classifier
    else:
        from ptclassifiernotr.TransformerSeqClassificationHandler import TransformersSeqClassifierHandler as Classifier
    ctxt = ModelContext()
    handler = Classifier()
    handler.initialize(ctxt)
    data = [{'data': 'To be or not to be, that is the question.'}]
    for i in range(1000):
        processed = handler.handle(data, ctxt)
        #print(processed)
    for m in ctxt.metrics.store:
        print(f'{m.name}: {m.value} {m.unit}')


if __name__ == '__main__':
    main()
