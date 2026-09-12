/* Generated from contracts/schemas. Run npm run contracts:generate; do not edit. */

/**
 * Reviewed learning content is independent of model results. Synthetic content is not an anatomical answer key.
 */
export interface Lesson {
  schemaVersion: '1.0.0';
  id: string;
  title: string;
  description: string;
  disclaimer: string;
  media: {
    id: string;
    kind: 'image_sequence' | 'video';
    width: number;
    height: number;
    src?: string;
  };
  /**
   * @minItems 1
   */
  checkpoints: [
    {
      id: string;
      frameNumber: number;
      timestampMs: number;
      imagePath: string;
      frameResultPath: string;
      /**
       * @minItems 1
       */
      questions: [
        {
          id: string;
          kind: 'anatomy_identification' | 'cvs_assessment';
          prompt: string;
          /**
           * @minItems 2
           */
          choices: [
            {
              id: string;
              label: string;
            },
            {
              id: string;
              label: string;
            },
            ...{
              id: string;
              label: string;
            }[]
          ];
          reviewedAnswer:
            | {
                choiceId: string;
                explanation: string;
                source: 'synthetic_mock';
                reviewerId?: string;
              }
            | {
                choiceId: string;
                explanation: string;
                source: 'expert_reviewed';
                reviewerId: string;
              };
        },
        ...{
          id: string;
          kind: 'anatomy_identification' | 'cvs_assessment';
          prompt: string;
          /**
           * @minItems 2
           */
          choices: [
            {
              id: string;
              label: string;
            },
            {
              id: string;
              label: string;
            },
            ...{
              id: string;
              label: string;
            }[]
          ];
          reviewedAnswer:
            | {
                choiceId: string;
                explanation: string;
                source: 'synthetic_mock';
                reviewerId?: string;
              }
            | {
                choiceId: string;
                explanation: string;
                source: 'expert_reviewed';
                reviewerId: string;
              };
        }[]
      ];
    },
    ...{
      id: string;
      frameNumber: number;
      timestampMs: number;
      imagePath: string;
      frameResultPath: string;
      /**
       * @minItems 1
       */
      questions: [
        {
          id: string;
          kind: 'anatomy_identification' | 'cvs_assessment';
          prompt: string;
          /**
           * @minItems 2
           */
          choices: [
            {
              id: string;
              label: string;
            },
            {
              id: string;
              label: string;
            },
            ...{
              id: string;
              label: string;
            }[]
          ];
          reviewedAnswer:
            | {
                choiceId: string;
                explanation: string;
                source: 'synthetic_mock';
                reviewerId?: string;
              }
            | {
                choiceId: string;
                explanation: string;
                source: 'expert_reviewed';
                reviewerId: string;
              };
        },
        ...{
          id: string;
          kind: 'anatomy_identification' | 'cvs_assessment';
          prompt: string;
          /**
           * @minItems 2
           */
          choices: [
            {
              id: string;
              label: string;
            },
            {
              id: string;
              label: string;
            },
            ...{
              id: string;
              label: string;
            }[]
          ];
          reviewedAnswer:
            | {
                choiceId: string;
                explanation: string;
                source: 'synthetic_mock';
                reviewerId?: string;
              }
            | {
                choiceId: string;
                explanation: string;
                source: 'expert_reviewed';
                reviewerId: string;
              };
        }[]
      ];
    }[]
  ];
}
