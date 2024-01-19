import { db } from '@/lib/db';
import { chats } from '@/lib/db/schema';
import { auth } from '@clerk/nextjs';
import { redirect } from 'next/navigation';
import { eq } from "drizzle-orm";
import React from 'react'
import ChatSideBar from '@/components/ChatSideBar';
import PDFViewer from '@/components/PDFViewer';
import ChatComponent from '@/components/ChatComponent';
import { checkSubscription } from '@/lib/subscription';

type Props = {
    params: {
        chatId: string
    };
};

const ChatPage = async ({ params: { chatId } }: Props) => {
    const { userId } = await auth()

    if (!userId) {
        return redirect('/sign-in')
    }

    const _chats = await db.select().from(chats).where(eq(chats.userId, userId))
    if (!_chats) return redirect('/')

    if (!_chats.find(chat => chat.id == parseInt(chatId))) {
        return redirect('/');
    }

    const currentChat = _chats.find(chat => chat.id === parseInt(chatId))
    const isPro = await checkSubscription();

    return (
        <div className='flex max-h-screen overflow-scroll no-scrollbar'>
            <div className='flex w-full max-h-screen overflow-scroll no-scrollbar'>
                {/* chat sidebar */}
                <div className='flex-[i] max-w-xs no-scrollbar'>
                    <ChatSideBar chats={_chats} chatId={parseInt(chatId)} isPro={isPro} />
                </div>

                {/* pdf viewer */}
                <div className='max-h-screen p-4 overflow-scroll flex-[5] no-scrollbar'>
                    <PDFViewer pdf_url={currentChat?.pdfUrl || ""} />
                </div>

                {/* chat component */}
                <div className='flex-[3] border-l-4 border-l-slate-200 no-scrollbar'>
                    <ChatComponent chatId={parseInt(chatId)} />
                </div>
            </div>
        </div>
    )
}

export default ChatPage