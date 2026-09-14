// Run with: node --test ml/review/test_review_core.cjs
// Fixtures are deliberately synthetic and never bundled as anatomy proposals.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const test = require('node:test');

const template = fs.readFileSync(path.join(__dirname,'review.template.html'),'utf8');
const source = template.split('// REVIEW_CORE_START')[1].split('// REVIEW_CORE_END')[0];
const sandbox = {Uint8Array};
vm.createContext(sandbox);
vm.runInContext(source.slice(source.indexOf('\n')) + '\nthis.core = ReviewCore;',sandbox);
const core = sandbox.core;
const clone = value => JSON.parse(JSON.stringify(value));
const digest = character => character.repeat(64);
const pixels = Uint8Array.from([1,1,1,0,0, 1,0,1,0,1, 1,1,1,0,0, 0,0,0,0,0]);
const candidate = {id:'endoscapes-box-7',sourceAnnotationId:7,structureId:'cystic_duct',source:'model_generated',mask:core.encodeRLE(pixels,5,4),maskSha256:digest('a'),bboxXYWH:[0,0,4,3]};
const bundle = {formatVersion:'1.0.0',artifactType:'candidate_mask_bundle',bundleId:'synthetic-ui-test-only',classes:[{structureId:'cystic_duct',label:'Synthetic duct fixture',color:'#53d6c7'}],images:[{id:'1_10',videoId:'1',frameNumber:10,split:'train',width:5,height:4,imageSha256:digest('b'),candidates:[candidate]}]};
const index = core.indexBundle(bundle,digest('c'));
function review(decision = 'pending') {
  return {formatVersion:'1.0.0',artifactType:'candidate_mask_review',bundleId:bundle.bundleId,bundleSha256:digest('c'),reviewer:{name:'Synthetic test author',reviewScope:'technical'},exportedAt:'2026-09-13T01:00:00.000Z',decisions:[{candidateId:candidate.id,imageId:'1_10',imageSha256:digest('b'),proposalMaskSha256:digest('a'),decision,mask:clone(candidate.mask),notes:'',reviewedAt:decision === 'pending' ? null : '2026-09-13T00:59:00.000Z',reviewMilliseconds:1250}]};
}

test('Row-major RLE round-trips a hole and disconnected component exactly',() => {
  assert.deepEqual([...core.decodeRLE(core.encodeRLE(pixels,5,4),5,4)],[...pixels]);
  assert.equal(core.decodeRLE(candidate.mask,5,4)[6],0);
  assert.equal(core.decodeRLE(candidate.mask,5,4)[9],1);
  assert.deepEqual([...core.encodeRLE(pixels,5,4).counts].slice(0,2),[0,3]);
});
test('Empty and full masks retain explicit complete coverage',() => {
  assert.deepEqual([...core.encodeRLE(new Uint8Array(20),5,4).counts],[20]);
  assert.deepEqual([...core.encodeRLE(new Uint8Array(20).fill(1),5,4).counts],[0,20]);
});
test('Malformed runs, dimensions, and nonbinary pixels fail closed',() => {
  for (const counts of [[],[0],[19],[21],[0,0,20],[1,0,19],[true,19],[1.5,18.5],[-1,21],[20,1]]) {
    assert.throws(() => core.decodeRLE({...clone(candidate.mask),counts},5,4));
  }
  assert.throws(() => core.decodeRLE(candidate.mask,4,5));
  assert.throws(() => core.encodeRLE(Uint8Array.from([0,2]),2,1));
});
test('Partial reviews preserve author and keep absent candidates absent',() => {
  const value = review(); value.decisions = [];
  const validated = core.validateReview(value,index);
  assert.equal(validated.reviewer.name,value.reviewer.name);
  assert.equal(validated.reviewer.reviewScope,'technical');
  assert.equal(validated.decisions.size,0);
});
test('Accepted requires nonempty unchanged pixels; edited requires actual changes',() => {
  assert.equal(core.validateReview(review('accepted'),index).decisions.get(candidate.id).decision,'accepted');
  const changed = pixels.slice(); changed[6] = 1;
  const value = review('accepted'); value.decisions[0].mask = core.encodeRLE(changed,5,4);
  assert.throws(() => core.validateReview(value,index),/equal the original/);
  value.decisions[0].decision = 'edited';
  assert.equal(core.validateReview(value,index).decisions.get(candidate.id).mask[6],1);
  assert.throws(() => core.validateReview(review('edited'),index),/differ/);
  value.decisions[0].mask = core.encodeRLE(new Uint8Array(20),5,4);
  assert.throws(() => core.validateReview(value,index),/empty mask/);
});
test('Pending edited pixels stay unapproved and require a null decision date',() => {
  const value = review(); value.decisions[0].mask = core.encodeRLE(new Uint8Array(20),5,4);
  assert.equal(core.validateReview(value,index).decisions.get(candidate.id).decision,'pending');
  value.decisions[0].reviewedAt = value.exportedAt;
  assert.throws(() => core.validateReview(value,index),/null review date/);
});
test('Rejection and expert referral require a meaningful note',() => {
  for (const decision of ['rejected','needs_expert']) {
    const value = review(decision); value.decisions[0].notes = '  ';
    assert.throws(() => core.validateReview(value,index),/require a note/);
    value.decisions[0].notes = 'Synthetic test reason.';
    assert.equal(core.validateReview(value,index).decisions.get(candidate.id).notes,'Synthetic test reason.');
  }
});
test('Wrong bundle, image, proposal, unknown and duplicate candidates are rejected',() => {
  for (const [key,value] of [['bundleId','other'],['bundleSha256',digest('d')]]) {
    const input = review(); input[key] = value; assert.throws(() => core.validateReview(input,index));
  }
  for (const key of ['candidateId','imageId','imageSha256','proposalMaskSha256']) {
    const input = review(); input.decisions[0][key] = 'other'; assert.throws(() => core.validateReview(input,index));
  }
  const duplicate = review(); duplicate.decisions.push(clone(duplicate.decisions[0]));
  assert.throws(() => core.validateReview(duplicate,index));
});
test('Reviewer, duration, state, and exact field sets are validated',() => {
  for (const mutate of [
    value => { value.reviewer.name = ' '; },
    value => { value.reviewer.reviewScope = 'reviewed'; },
    value => { value.reviewer.credentials = 'fabricated'; },
    value => { value.extra = true; },
    value => { delete value.decisions[0].notes; },
    value => { value.decisions[0].reviewMilliseconds = true; },
    value => { value.decisions[0].reviewMilliseconds = NaN; },
    value => { value.decisions[0].reviewMilliseconds = Infinity; },
    value => { value.decisions[0].reviewMilliseconds = -1; },
    value => { value.decisions[0].decision = 'auto_approved'; }
  ]) { const input = review(); mutate(input); assert.throws(() => core.validateReview(input,index)); }
});
test('Reviewer control characters cannot be imported or assigned an author',() => {
  for (const character of ['\u0000','\u0009','\u000a','\u001f']) {
    const input = review(); input.reviewer.name = `Reviewer${character}Name`;
    assert.throws(() => core.validateReview(input,index),/named reviewer/);
  }
});
test('UTC dates accept Z and zero offset, reject malformed or impossible dates',() => {
  for (const value of ['2026-09-13T01:00:00Z','2026-09-13T01:00:00.123456Z','2026-09-13T01:00:00+00:00']) assert.equal(core.utc(value),true,value);
  for (const value of ['2026-02-31T01:00:00Z','2026-09-13','2026-09-13T01:00:00-04:00','2026-09-13T25:00:00Z','invalid']) assert.equal(core.utc(value),false,value);
});
test('Validation never mutates imported content or immutable proposal pixels',() => {
  const input = review('accepted'), before = JSON.stringify(input), original = [...index.entries.get(candidate.id).original];
  const validated = core.validateReview(input,index); validated.decisions.get(candidate.id).mask.fill(0);
  assert.equal(JSON.stringify(input),before);
  assert.deepEqual([...index.entries.get(candidate.id).original],original);
});
