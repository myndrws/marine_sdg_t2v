# this script is for etl of gov data
import requests
import io
from pypdf import PdfReader
from bs4 import BeautifulSoup
import pickle
import nltk.data
import time


class etl_data():
    """
    This class goes through every file one by one
    # and sentence-tokenize the whole file and save each sentence
    # so that you could link it back to the master_dict via a join if required
    # which can basically be done by appending another unique id for the position of the sentence
    # so you have the id e.g. 42_18, then the sentence_id, 1 to N to create any link.
    # Each sentence entry is therefore source_id, sentence_id, sentence.
    """

    def __init__(self, master_dict: dict):

        assert isinstance(master_dict, dict), "Expecting a dictionary for master_dict"

        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.html_attrs = [{'class': 'main-content-container'},
                           {'class': 'govspeak'},
                           {'id': 'contents'},
                           {'id': 'content'}]
        self.uids = []
        self.sentences = []
        report_count = 0

        print(f"Starting at {time.ctime()}")

        for source_id, entry in master_dict.items():
            if 'first_content_url' not in entry:
                print(f"Skipping entry {source_id} at loop {report_count} (no content url in master_dict).")
                continue
            print(f"Unpacking entry {source_id} at loop {report_count}.")
            self.source_id = source_id
            self.entry_url = entry['first_content_url']   # we assume all are pdfs or html urls
            self.full_text = self._extract_from_html_and_pdf()
            self.sent_token_full_text = self._sent_tokenize_docs()
            self._append_uids_sentences()
            report_count += 1

        self._reset_self()
        print(f"Finished unpacking! Time is {time.ctime()}")

    def _extract_from_pdf(self):
        # assume you pass in a pdf url
        response = requests.get(self.entry_url)
        with io.BytesIO(response.content) as open_pdf_file:
            reader = PdfReader(open_pdf_file)
            all_pages = '. '.join((page.extract_text() for page in reader.pages))
        return all_pages

    def _extract_from_html(self):
        # assume you pass in an html url
        response = requests.get(self.entry_url)
        parsed_html = BeautifulSoup(response.text, features="lxml")
        for attrs in self.html_attrs:
            scan_result = parsed_html.body.find('div', attrs=attrs)
            if scan_result is not None and hasattr(scan_result, 'text'):
                return parsed_html.body.find('div', attrs=attrs).text
        raise ValueError(f"None of the searched attributes can locate html "
                         f"body text content for document {self.source_id}")

    def _extract_from_html_and_pdf(self):
        if self.entry_url.endswith((".PDF", ".pdf")):
            return self._extract_from_pdf()
        else:
            return self._extract_from_html()

    def _sent_tokenize_docs(self):
        return self.sent_tokenizer.tokenize(self.full_text.replace('\n', '').replace('.', '. ').strip())

    def _append_uids_sentences(self):
        for id, sent in enumerate(self.sent_token_full_text):
            # heuristic for checking there are more than two words in the sentence
            # without having to tokenize the sentence to words first (keeps it fast)
            if sent.count(' ') > 2:
                self.uids.append(self.source_id + f'_{id}')
                self.sentences.append(sent)

    def _reset_self(self):
        self.source_id = None
        self.entry = None
        self.full_text = None
        self.sent_token_full_text = None


if __name__ == '__main__':

    master_dict_filename = 'master_dict_2023-05-05.pkl'
    with open(master_dict_filename, 'rb') as f:
        master_dict = pickle.load(f)

    data = etl_data(master_dict)

    filename = f"sentences_from_{master_dict_filename}"
    print(filename)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    # look at one data source that is an html format
    # first_content_url = master_dict['4_6']['first_content_url']
    # page = requests.get(first_content_url)
    # parsed_html = BeautifulSoup(page.text, features="lxml")
    # found = parsed_html.body.find('div', attrs={'class': 'main-content-container'})
    #
    # # look at one data source that is a pdf format
    # first_content_url = master_dict['42_18']['first_content_url']
    # response = requests.get(first_content_url)
    # with io.BytesIO(response.content) as open_pdf_file:
    #     reader = PdfReader(open_pdf_file)
    #     all_pages = '. '.join((page.extract_text() for page in reader.pages))


