import { AIProvider, InferenceConfig } from "@/ai";
import config from "../config";
import { info } from "@actions/core";
import { generateObject, jsonSchema } from "ai";
import { zodToJsonSchema } from "zod-to-json-schema";

export class AISDKProvider implements AIProvider {
  private createAiFunc: any;
  private modelName: string;

  constructor(createAiFunc: any, modelName: string) {
    this.createAiFunc = createAiFunc;
    this.modelName = modelName;
  }

  async runInference({
    prompt,
    temperature,
    system,
    schema,
  }: InferenceConfig): Promise<any> {
    const llm = this.createAiFunc({
      apiKey: config.llmApiKey,
    });

    // Convert Zod schema to JSON schema and use it directly
    // This bypasses the AI SDK's internal schema wrapping that causes
    // the $PARAMETER_NAME issue with some providers
    const jsonSchemaObj = zodToJsonSchema(schema);

    const { object, usage } = await generateObject({
      model: llm(this.modelName),
      prompt,
      temperature: temperature || 0,
      system,
      schema: jsonSchema(jsonSchemaObj as any),
    });

    if (process.env.DEBUG) {
      info(`usage: \n${JSON.stringify(usage, null, 2)}`);
    }

    // Handle case where response is wrapped under $PARAMETER_NAME
    if (object && typeof object === 'object' && '$PARAMETER_NAME' in object) {
      const unwrapped = (object as any)['$PARAMETER_NAME'];
      // Validate against the original Zod schema
      return schema.parse(unwrapped);
    }

    return object;
  }
}
