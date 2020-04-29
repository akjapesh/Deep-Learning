
import os

from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser

corpusroot = './docs'


class Index(object):
    '''Integrate whoosh's indexer and searcher'''

    def __init__(self, index_path, schema=None):
        if not os.path.exists(index_path):
            if schema is None:
                raise ValueError('You need to specify a `schema` when creating'
                                 ' an index')
            os.mkdir(index_path)
            self._index = create_in(index_path, schema)
            self._schema = schema
        else:
            self._index = open_dir(index_path)
            self._schema = self._index.schema

    def add_document(self, **document):
        '''Add a document to the index.
        Document properties should be passed as parameters, like in:
        >>> my_index.add_document(title=u'My Title', content=u'The content')
        '''
        writer = self._index.writer()
        writer.add_document(**document)
        writer.commit()

    def add_documents(self, documents):
        '''Add a list of documents (`list` of `dict`s) to the index
        It's an optimized version of `add_document` since it calls `commit`
        only in the end.
        '''
        writer = self._index.writer()
        for document in documents:
            writer.add_document(**document)
        writer.commit()

    def search(self, query, field):
        query_object = QueryParser(field, self._schema).parse(query)
        searcher = self._index.searcher()
        results = searcher.search(query_object)
        return results


if __name__ == '__main__':
    import shutil


    index_path = 'index-test'
    try:
        shutil.rmtree(index_path)
    except OSError:
        pass

    schema = Schema(filename=TEXT(stored=True), id=ID(stored=True), content=TEXT)
    my_index = Index(index_path, schema)

    doc_1 = {u'filename': u'a.txt', u'id': u'1', u'content': u'first document'}
    doc_2 = {u'filename': u'b.txt', u'id': u'2', u'content': u'2nd document'}
    doc_3 = {u'filename': u'c.txt', u'id': u'3', u'content': u'3rd document'}
    documents = [doc_1, doc_2, doc_3]
    my_index.add_documents(documents)

    result = my_index.search(u'first', u'content')
    # assert len(result) == 1
    # assert result[0][u'id'] == u'1'

    doc_4 = {u'filename': u'a.txt', u'id': u'4', u'content': u'not first'}
    my_index.add_document(**doc_4)

    result = my_index.search(u'first', u'content')
    # assert len(result) == 2
    # assert set([doc[u'id'] for doc in result]) == set([u'1', u'4'])


    i=0
    doccs=[]
    for filename in os.listdir(corpusroot):
        i+=1
        file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
        docy = file.read()
        file.close()
        docc={u'filename':str(filename),u'id':str(i),u'content':str(docy)}
        doccs+=[docc]
    my_index.add_documents(doccs)
    result=my_index.search(u'health insurance wall street',u'content')
    print(result[0])
