declare module "js-aruco2" {
  export const AR: {
    Detector: new (options: { dictionaryName: string; maxHammingDistance?: number }) => { detect(image: ImageData): { id: number; corners: { x: number; y: number }[] }[] };
    Dictionary: new (name: string) => { generateSVG(id: number): string; codeList: string[] };
  };
  const aruco: { AR: typeof AR };
  export default aruco;
}
declare module "js-aruco2/src/posit1.js" {
  export const POS: { Posit: new (size: number, focal: number) => { pose(corners: { x: number; y: number }[]): {
    bestError: number; bestRotation: number[][]; bestTranslation: number[];
    alternativeError: number; alternativeRotation: number[][]; alternativeTranslation: number[];
  } } };
  const posit: { POS: typeof POS };
  export default posit;
}
