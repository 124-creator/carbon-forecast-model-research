import unittest

from event_data import parse_article


class DataTests(unittest.TestCase):
    def test_visible_article_header_without_meta_has_literal_publication(self):
        text = ('<h1>2020年12月上旬流通领域重要生产资料市场价格变动情况</h1>'
                '<div class="detail-title-des"><h2><p>2020/12/14 09:30</p>'
                '<span>来源：国家统计局</span></h2></div><table>'
                '<tr><td>普通混煤（4500大卡）</td><td>吨</td><td>401.2</td></tr>'
                '<tr><td>山西大混（5000大卡）</td><td>吨</td><td>500.3</td></tr>'
                '<tr><td>山西优混（5500大卡）</td><td>吨</td><td>590.1</td></tr></table>')
        event = parse_article(text)
        self.assertIsNotNone(event)
        self.assertEqual(event['available_date'], '2020-12-15')

    def test_literal_thermal_vector_is_published_next_day(self):
        text = ('<meta name="ArticleTitle" content="2019年12月上旬流通领域重要生产资料市场价格变动情况">'
                '<meta name="PubDate" content="2019/12/16 09:30">'
                '<meta name="ContentSource" content="国家统计局">'
                '<p>2019/12/16 09:30</p><table>'
                '<tr><td>普通混煤（4500 大卡）</td><td>吨</td><td>401.2</td></tr>'
                '<tr><td>山西大混（5000大卡）</td><td>吨</td><td>500.3</td></tr>'
                '<tr><td>山西优混（5500大卡）</td><td>吨</td><td>590.1</td></tr></table>')
        event = parse_article(text)
        self.assertEqual(event['available_date'], '2019-12-17')
        self.assertEqual(event['values'], [401.2, 500.3, 590.1])
        self.assertEqual(event['labels'][0], '普通混煤（4500 大卡）')
        self.assertIsNone(parse_article(text.replace('name="PubDate"', 'name="WrittenDate"')))


if __name__ == '__main__':
    unittest.main()
