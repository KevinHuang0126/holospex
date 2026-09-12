// JSON Schema is the wire-format authority. Never hand-edit generated TS types.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { compile } from 'json-schema-to-typescript';

const root = new URL('../', import.meta.url);
const check = process.argv.includes('--check');
await mkdir(new URL('contracts/src/generated/', root), { recursive: true });
for (const name of ['lesson', 'frame-result', 'learner-attempt']) {
  const schema = JSON.parse(await readFile(new URL(`contracts/schemas/${name}.schema.json`, root), 'utf8'));
  const output = await compile(schema, schema.title, {
    bannerComment: '/* Generated from contracts/schemas. Run npm run contracts:generate; do not edit. */',
    additionalProperties: false,
    style: { singleQuote: true },
  });
  const destination = new URL(`contracts/src/generated/${name}.ts`, root);
  if (check) {
    const current = await readFile(destination, 'utf8').catch(() => '');
    if (current !== output) throw new Error(`Stale ${name} types. Run npm run contracts:generate.`);
  } else await writeFile(destination, output);
}
console.log(check ? 'Contract types match schemas.' : 'Contract types generated.');
