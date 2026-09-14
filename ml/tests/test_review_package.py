"""Verify portable review packaging preserves source identity and initial state."""
import hashlib
import importlib.util
import json
import copy
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

from PIL import Image

REVIEW_DIR = Path(__file__).resolve().parents[1] / 'review'
sys.path.insert(0, str(REVIEW_DIR))
SPEC = importlib.util.spec_from_file_location('package_review', REVIEW_DIR / 'package_review.py')
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


def fixture(root):
    image_path=root/'images/1_100.jpg';image_path.parent.mkdir()
    Image.new('RGB',(3,2),'#445566').save(image_path)
    pixels=bytes([0,1,0,0,1,0])
    bundle={'formatVersion':'1.0.0','artifactType':'candidate_mask_bundle','bundleId':'synthetic-package-test','createdAt':'2026-09-13T00:00:00Z','sourceManifestSha256':'1'*64,
      'selection':{'rule':'synthetic test fixture'},'generator':{'name':'synthetic test fixture </script><script>unexpected()</script>'},
      'classes':[{'structureId':'cystic_artery','label':'Cystic artery','color':'#f08080'}],
      'images':[{'id':'1_100','videoId':'1','frameNumber':100,'split':'train','width':3,'height':2,'imagePath':'images/1_100.jpg','imageSha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),
       'candidates':[{'id':'endoscapes-box-100','sourceAnnotationId':100,'structureId':'cystic_artery','bboxXYWH':[1,0,1,2],'source':'model_generated','mask':{'encoding':'rle-row-major-v1','width':3,'height':2,'counts':[1,1,2,1,1]},'maskSha256':hashlib.sha256(pixels).hexdigest(),'proposalScore':.5,'scoreMeaning':'Synthetic score; not anatomy confidence','warnings':[]}]}]}
    path=root/'bundle.json';path.write_text(json.dumps(bundle,indent=2)+'\n')
    template=root/'template.html';template.write_text('<!doctype html><script>window.HOLOSPEX_REVIEW=__HOLOSPEX_REVIEW_DATA__;</script>')
    return path,template


class ReviewPackageTests(unittest.TestCase):
    def test_reviewer_instructions_match_batch_and_are_verified_in_zip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);bundle,template=fixture(root)
            data=json.loads(bundle.read_text())
            data['classes'].append({'structureId':'cystic_duct','label':'Cystic duct','color':'#ffd166'})
            # Distinct image, case and mask totals catch hard-coded pilot counts.
            for frame in (200,300):
                row=copy.deepcopy(data['images'][0])
                row.update(id=f'2_{frame}',videoId='2',frameNumber=frame,imagePath=f'images/2_{frame}.jpg')
                (root/row['imagePath']).write_bytes((root/'images/1_100.jpg').read_bytes())
                row['candidates'][0].update(id=f'endoscapes-box-{frame}',sourceAnnotationId=frame)
                data['images'].append(row)
            extra=copy.deepcopy(data['images'][-1]['candidates'][0])
            extra.update(id='endoscapes-box-301',sourceAnnotationId=301,structureId='cystic_duct')
            data['images'][-1]['candidates'].append(extra)
            bundle.write_text(json.dumps(data,indent=2)+'\n')
            original=bundle.read_bytes()
            out=root/'next-batch';report=package.build_package(bundle,out,template_path=template)

            instructions=(out/'START-HERE.txt').read_text()
            self.assertIn('3 images from 2 surgical cases; 4 masks to review.',instructions)
            self.assertIn('"4 / 4 decisions recorded"',instructions)
            self.assertIn('Cystic artery, Cystic duct',instructions)
            self.assertIn('extract/unzip it first',instructions)
            self.assertIn('Send ONLY that newest .json file',instructions)
            self.assertIn('2 anatomy classes',(out/'README.md').read_text())
            self.assertNotIn('pilot',(out/'README.md').read_text().lower())
            # Every named action must use an existing button label, including
            # the easily confused edited-approval and JSON-resume actions.
            live_template=(REVIEW_DIR/'review.template.html').read_text()
            for label in ('Start review','Accept unchanged','Approve edit','Reject','Needs expert',
                          'Export review JSON','Resume from JSON','Restore saved review'):
                self.assertIn(f'"{label}"',instructions)
                self.assertIn(f'>{label}</button>',live_template)

            expected=(out/'START-HERE.txt').read_bytes()
            inventory_entry=next(row for row in report['files'] if row['path']=='START-HERE.txt')
            self.assertEqual(inventory_entry['bytes'],len(expected))
            self.assertEqual(inventory_entry['sha256'],hashlib.sha256(expected).hexdigest())
            manifest=json.loads((out/'handoff-manifest.json').read_text())
            self.assertIn(inventory_entry,manifest['files'])
            self.assertEqual((report['imageCount'],report['caseCount'],report['candidateCount']),(3,2,4))
            with zipfile.ZipFile(out.with_suffix('.zip')) as z:
                self.assertEqual(z.read('next-batch/START-HERE.txt'),expected)
                self.assertEqual(z.read('next-batch/bundle.json'),original)
                self.assertIsNone(z.testzip())
            self.assertEqual(bundle.read_bytes(),original)
            self.assertEqual((out/'bundle.json').read_bytes(),original)
            self.assertEqual(report['bundleSha256'],hashlib.sha256(original).hexdigest())
            self.assertEqual(report['reviewDecisionsCreated'],0)

    def test_single_html_embeds_original_data_and_zip_preserves_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);bundle,template=fixture(root);original=bundle.read_bytes()
            out=root/'handoff';report=package.build_package(bundle,out,template_path=template)
            self.assertEqual(report['reviewDecisionsCreated'],0)
            self.assertEqual((out/'bundle.json').read_bytes(),original)
            html=(out/'review.html').read_text()
            self.assertIn('data:image/jpeg;base64,',html)
            self.assertNotIn('</script><script>unexpected()',html)
            self.assertEqual(html.count('</script>'),1)
            self.assertNotIn('__HOLOSPEX_REVIEW_DATA__',html)
            with zipfile.ZipFile(out.with_suffix('.zip')) as z:
                self.assertEqual(z.read('handoff/bundle.json'),original)
                self.assertEqual(z.read('handoff/images/1_100.jpg'),(root/'images/1_100.jpg').read_bytes())
                self.assertIsNone(z.testzip())
            with self.assertRaises(ValueError):package.build_package(bundle,out,template_path=template)

    def test_changed_image_rejects_before_exposing_any_handoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);bundle,template=fixture(root)
            Image.new('RGB',(3,2),'red').save(root/'images/1_100.jpg')
            out=root/'handoff'
            with self.assertRaisesRegex(ValueError,'hash'):package.build_package(bundle,out,template_path=template)
            self.assertFalse(out.exists());self.assertFalse(out.with_suffix('.zip').exists())

    def test_wrong_dimensions_and_path_escape_are_not_packaged(self):
        for mutate in ['dimensions','path']:
            with self.subTest(mutate=mutate),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);bundle,template=fixture(root);data=json.loads(bundle.read_text())
                if mutate=='dimensions':
                    Image.new('RGB',(4,2),'#445566').save(root/'images/1_100.jpg')
                    data['images'][0]['imageSha256']=hashlib.sha256((root/'images/1_100.jpg').read_bytes()).hexdigest()
                else:data['images'][0]['imagePath']='../outside.jpg'
                bundle.write_text(json.dumps(data));out=root/'handoff'
                with self.assertRaises(ValueError):package.build_package(bundle,out,template_path=template)
                self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
