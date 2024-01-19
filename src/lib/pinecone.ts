import { Pinecone, PineconeRecord } from "@pinecone-database/pinecone";
import { downloadFromS3 } from "./s3-server";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import {
  Document,
  RecursiveCharacterTextSplitter,
} from "@pinecone-database/doc-splitter";
import { getEmbeddings } from "./embeddings";
import md5 from "md5";
import { convertToAsci } from "./utils";

let pinecone: Pinecone | null = null;

export const getPineConClient = async () => {
  if (!pinecone) {
    pinecone = new Pinecone({
      environment: process.env.PINECONE_ENV!,
      apiKey: process.env.PINECONE_API_KEY!,
    });
  }

  return pinecone;
};

type PDFPage = {
  pageContent: string;
  metadata: {
    loc: { pageNumber: number };
  };
};

export async function loadS3IntoPinecone(file_key: string) {
  //1. obtain the pdf -> download and read from pdf
  console.log("downloading s3 into file system");
  const file_name = await downloadFromS3(file_key);

  if (!file_name) {
    throw new Error("could not download from s3");
  }
  const loader = new PDFLoader(file_name);
  const pages = (await loader.load()) as PDFPage[];

  //2. split and segment the pdf
  const documents = await Promise.all(pages.map(prepareDocument));

  //3. vectorise and embed individual documents
  const vectors = await Promise.all(documents.flat().map(embedDocument));

  //4. upload to pinecone
  const client = await getPineConClient();
  const pineconeIndex = client.Index("chatpdf-online");

  console.log("inserting vectors into pinecone");
  const namespace = convertToAsci(file_key);
  //chunkedUpsert: (index: VectorOperationsApi, vectors: Vector[], namespace: string, chunkSize?: number) => Promise<boolean>;
  //console.log(vectors);
  pineconeIndex.namespace(namespace).upsert(vectors);

  // for (const ids_vectors_chunk of chunks(vectors, 10)) {
  //   const records = ids_vectors_chunk.map((vector) => ({
  //     id: vector.id,
  //     values: vector.values,
  //     metadata: vector.metadata,
  //   }));

  //   try {
  //     await pineconeIndex.namespace(namespace).upsert(records);
  //   } catch (error) {
  //     console.error("Error during upsert:", error);
  //     throw error; // Handle or log the error as needed
  //   }
  // }

  return documents[0];
}

function* chunks<T>(
  iterable: Iterable<T>,
  batch_size: number = 100
): Generator<T[], void, unknown> {
  // A helper function to break an iterable into chunks of size batch_size
  let it = iterable[Symbol.iterator]();
  let chunk = Array.from({ length: batch_size }, () => it.next().value) as T[];
  while (chunk[0] !== undefined) {
    yield chunk;
    chunk = Array.from({ length: batch_size }, () => it.next().value) as T[];
  }
}

async function embedDocument(doc: Document) {
  try {
    const embeddings = await getEmbeddings(doc.pageContent);
    const hash = md5(doc.pageContent);

    return {
      id: hash,
      values: embeddings,
      metadata: {
        text: doc.metadata.text,
        pageNumber: doc.metadata.pageNumber,
      },
    } as PineconeRecord;
  } catch (error) {
    console.log("error embedding document", error);
    throw error;
  }
}

export const truncateStringByByte = (str: string, bytes: number) => {
  const enc = new TextEncoder();
  return new TextDecoder("utf-8").decode(enc.encode(str).slice(0, bytes));
};

async function prepareDocument(page: PDFPage) {
  let { pageContent, metadata } = page;
  pageContent = pageContent.replace(/\n/g, "");
  //split the docs
  const splitter = new RecursiveCharacterTextSplitter();
  const docs = await splitter.splitDocuments([
    new Document({
      pageContent,
      metadata: {
        pageNumber: metadata.loc.pageNumber,
        text: truncateStringByByte(pageContent, 36000),
      },
    }),
  ]);
  return docs;
}
